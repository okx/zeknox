#[repr(C)]
pub struct Error {
    pub code: i32,
    str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
}

impl Drop for Error {
    fn drop(&mut self) {
        extern "C" {
            fn free(str: Option<core::ptr::NonNull<i8>>);
        }
        unsafe { free(self.str) };
        self.str = None;
    }
}

impl From<&Error> for String {
    fn from(status: &Error) -> Self {
        if let Some(str) = status.str {
            let c_str = unsafe { std::ffi::CStr::from_ptr(str.as_ptr() as *const _) };
            String::from(c_str.to_str().unwrap_or("unintelligible"))
        } else {
            format!("sppark::Error #{}", status.code)
        }
    }
}

impl From<Error> for String {
    fn from(status: Error) -> Self {
        String::from(&status)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", String::from(self))
    }
}