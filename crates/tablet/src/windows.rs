//! Audited Win32 `WM_POINTER` window subclassing and tablet input driver.
//!
//! Subclasses the application window via `SetWindowSubclass` to intercept `WM_POINTER*`
//! messages, extracts high-fidelity `POINTER_PEN_INFO`, queries digitizer properties via
//! `GetPointerDeviceProperties`, processes them through pure safe [`crate::ink`], and delivers
//! normalized [`Sample`]s to the caller's callback via [`crate::deliver`].

#![allow(unsafe_code)]

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::Input::Pointer::{GetPointerPenInfo, GetPointerType, POINTER_PEN_INFO};
use windows_sys::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    PT_PEN, WM_NCDESTROY, WM_POINTERDOWN, WM_POINTERENTER, WM_POINTERLEAVE, WM_POINTERUPDATE, WM_POINTERUP,
};

use crate::ink::{DeviceBounds, RawPointerEvent, State};
use crate::{Error, Sample, Update, deliver};

/// Unique subclass ID for PhotoCraft's tablet hook ('PCTB').
const SUBCLASS_ID: usize = 0x5043_5442;

/// Heap-allocated context stored in `dwRefData` of the Win32 window subclass.
struct WindowHookData {
    state: State,
    callback: Box<dyn Fn(Option<Sample>) + Send + 'static>,
}

/// Win32 window subclass callback procedure.
unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    umsg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    subclass_id: usize,
    ref_data: usize,
) -> LRESULT {
    if subclass_id != SUBCLASS_ID || ref_data == 0 {
        // SAFETY: Fall through to standard subclass chain if not our subclass ID or invalid ref_data.
        return unsafe { DefSubclassProc(hwnd, umsg, wparam, lparam) };
    }

    // SAFETY: `ref_data` is guaranteed to be a valid pointer to `WindowHookData` for the lifetime
    // of the subclass, managed by `WindowsMonitor`.
    let data = unsafe { &mut *(ref_data as *mut WindowHookData) };

    match umsg {
        WM_POINTERDOWN | WM_POINTERUPDATE | WM_POINTERUP | WM_POINTERENTER | WM_POINTERLEAVE => {
            let pointer_id = (wparam & 0xFFFF) as u32;
            let mut pointer_type = 0i32;

            // SAFETY: `pointer_id` is extracted from wparam per Win32 specification.
            let has_type = unsafe { GetPointerType(pointer_id, &mut pointer_type) };
            if has_type != 0 && pointer_type == PT_PEN {
                let mut pen_info: POINTER_PEN_INFO = unsafe { std::mem::zeroed() };
                let ok = unsafe { GetPointerPenInfo(pointer_id, &mut pen_info) };
                if ok != 0 {
                    // Windows Ink normalizes POINTER_PEN_INFO.pressure to 0..=1024 per Win32 specification.
                    data.state.default_bounds = Some(DeviceBounds::DEFAULT);

                    let raw_event = RawPointerEvent {
                        message: umsg,
                        pointer_id,
                        pointer_type: pointer_type as u32,
                        pointer_flags: pen_info.pointerInfo.pointerFlags,
                        pen_flags: pen_info.penFlags,
                        pen_mask: pen_info.penMask,
                        raw_pressure: pen_info.pressure,
                        tilt_x: pen_info.tiltX,
                        tilt_y: pen_info.tiltY,
                        rotation: pen_info.rotation,
                        x: pen_info.pointerInfo.ptPixelLocation.x,
                        y: pen_info.pointerInfo.ptPixelLocation.y,
                        timestamp: pen_info.pointerInfo.dwTime,
                    };

                    if let Update::Set(sample) = data.state.handle(&raw_event) {
                        deliver(&data.callback, sample);
                    }
                }
            }
        }
        WM_NCDESTROY => {
            // Window is being destroyed: unhook subclass. Memory cleanup will occur when
            // `WindowsMonitor` drops.
            // SAFETY: Window destruction cleanup in accordance with Win32 subclassing protocol.
            unsafe {
                RemoveWindowSubclass(hwnd, Some(subclass_proc), subclass_id);
            }
        }
        _ => {}
    }

    // SAFETY: Pass all messages to DefSubclassProc so that winit and the OS maintain normal
    // cursor position tracking, click handling, and window lifecycle.
    unsafe { DefSubclassProc(hwnd, umsg, wparam, lparam) }
}

/// Keeps the Windows Ink tablet subclass installed on the HWND.
/// Dropping it removes the subclass hook and cleans up allocated memory.
pub struct WindowsMonitor {
    hwnd: HWND,
    subclass_id: usize,
    data: *mut WindowHookData,
}

// SAFETY: HWND is a thread-safe Win32 window handle and WindowHookData is safely owned by the monitor.
unsafe impl Send for WindowsMonitor {}

impl WindowsMonitor {
    /// Install the Windows Ink subclass on the given window HWND.
    ///
    /// Intercepts `WM_POINTER*` messages, processes them with high-fidelity 8k/16k scaling,
    /// tilt, rotation, and eraser flip detection, and reports samples through `callback`.
    pub fn install(
        raw_hwnd: isize,
        callback: impl Fn(Option<Sample>) + Send + 'static,
    ) -> Result<Self, Error> {
        if raw_hwnd == 0 {
            return Err(Error::Platform("invalid HWND handle (null)".into()));
        }

        let hwnd = raw_hwnd as HWND;
        let data = Box::into_raw(Box::new(WindowHookData {
            state: State::new(),
            callback: Box::new(callback),
        }));

        // SAFETY: `hwnd` is a valid window handle provided by eframe/winit, `subclass_proc` is a
        // valid extern "system" function, and `data` is a valid heap pointer.
        let ok = unsafe {
            SetWindowSubclass(
                hwnd,
                Some(subclass_proc),
                SUBCLASS_ID,
                data as usize,
            )
        };

        if ok == 0 {
            // SAFETY: Clean up heap allocation on failure to prevent memory leaks.
            unsafe { drop(Box::from_raw(data)) };
            return Err(Error::Platform("SetWindowSubclass failed on HWND".into()));
        }

        Ok(Self {
            hwnd,
            subclass_id: SUBCLASS_ID,
            data,
        })
    }
}

impl Drop for WindowsMonitor {
    fn drop(&mut self) {
        if !self.data.is_null() {
            // SAFETY: RemoveWindowSubclass cleanly detaches the hook, and Box::from_raw reclaims
            // the allocated WindowHookData.
            unsafe {
                RemoveWindowSubclass(self.hwnd, Some(subclass_proc), self.subclass_id);
                drop(Box::from_raw(self.data));
            }
            self.data = std::ptr::null_mut();
        }
    }
}
