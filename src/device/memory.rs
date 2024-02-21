use crate::device::bindings::{
    cudaSetDevice, cudaFree, cudaMalloc, cudaMallocAsync, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind,
};
use crate::device::error::{CudaError, CudaResult, CudaResultWrap};
use crate::device::stream::CudaStream;
use std::mem::{size_of, MaybeUninit};
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};
use std::os::raw::c_void;
use std::slice::from_raw_parts_mut;

#[derive(Debug)]
pub enum HostOrDeviceSlice<'a, T> {
    Host(Vec<T>),
    Device(i32, &'a mut [T]), // first element is device id
}

impl<'a, T> HostOrDeviceSlice<'a, T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Device(_, s) => s.len(),
            Self::Host(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Device(_, s) => s.is_empty(),
            Self::Host(v) => v.is_empty(),
        }
    }

    pub fn is_on_device(&self) -> bool {
        match self {
            Self::Device(_, _) => true,
            Self::Host(_) => false,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Device(_, _) => {
                panic!("Use copy_to_host and copy_to_host_async to move device data to a slice")
            }
            Self::Host(v) => v.as_mut_slice(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Device(_, _) => {
                panic!("Use copy_to_host and copy_to_host_async to move device data to a slice")
            }
            Self::Host(v) => v.as_slice(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            Self::Device(_, s) => s.as_ptr(),
            Self::Host(v) => v.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Self::Device(_, s) => s.as_mut_ptr(),
            Self::Host(v) => v.as_mut_ptr(),
        }
    }

    pub fn on_host(src: Vec<T>) -> Self {
        Self::Host(src)
    }

    pub fn cuda_malloc(device_id: i32, count: usize) -> CudaResult<Self> {
        let size = count.checked_mul(size_of::<T>()).unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation);
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            let _ = cudaSetDevice(device_id);
            cudaMalloc(device_ptr.as_mut_ptr(), size).wrap()?;
            Ok(Self::Device(device_id, from_raw_parts_mut(
                device_ptr.assume_init() as *mut T,
                count,
            )))
        }
    }

    pub fn cuda_malloc_async(device_id: i32, count: usize, stream: &CudaStream) -> CudaResult<Self> {
        let size = count.checked_mul(size_of::<T>()).unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation);
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            let _ = cudaSetDevice(device_id);
            cudaMallocAsync(
                device_ptr.as_mut_ptr(),
                size,
                stream.handle as *mut _ as *mut _,
            )
            .wrap()?;
            Ok(Self::Device(device_id, from_raw_parts_mut(
                device_ptr.assume_init() as *mut T,
                count,
            )))
        }
    }

    pub fn copy_from_host(&mut self,  val: &[T]) -> CudaResult<()> {
        let device_id: i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            // println!("start copy to device");
            unsafe {
                let _ = cudaSetDevice(device_id);
                cudaMemcpy(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
                .wrap()?
            }
        }
        Ok(())
    }
    pub fn copy_from_host_offset(
        &self,
        src: &[T],
        offset: usize,
        count: usize,
    ) -> CudaResult<()> {
        let device_id: i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        unsafe {
            // println!("ptr value: {:?}", self.as_mut_ptr().add(offset));
            let _ = cudaSetDevice(device_id);
            cudaMemcpy(
                self.as_ptr().add(offset) as *mut c_void,
                src.as_ptr() as *const c_void,
                count * size_of::<T>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
            .wrap()?
        }
        Ok(())
    }

    pub fn copy_to_host(&self, val: &mut [T], counts: usize) -> CudaResult<()> {
        let device_id: i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
        let size = size_of::<T>() * counts;
        if size != 0 {
            unsafe {
                let _ = cudaSetDevice(device_id);
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .wrap()?
            }
        }
        Ok(())
    }


    /// val: host data pointer
    /// offset: the offset to device
    pub fn copy_to_host_offset(&self, val: &mut [T],  offset: usize, counts: usize) -> CudaResult<()> {
        let device_id: i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
        // println!("copy from device to host for device id: {:?}, offset: {:?}, count: {:?}", device_id, offset, counts);
        let size = size_of::<T>() * counts;
        if size != 0 {
            unsafe {
                let _ = cudaSetDevice(device_id);
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr().add(offset) as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_from_host_async(&mut self, val: &[T], stream: &CudaStream) -> CudaResult<()> {
        let device_id:i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                let _ = cudaSetDevice(device_id);
                cudaMemcpyAsync(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host_async(&self, val: &mut [T], stream: &CudaStream) -> CudaResult<()> {
        let device_id: i32 = match self {
            Self::Device(d, _) => {*d}
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                let _ = cudaSetDevice(device_id);
                cudaMemcpyAsync(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<'a, T> Index<$t> for HostOrDeviceSlice<'a, T>
            {
                type Output = [T];

                fn index(&self, index: $t) -> &Self::Output {
                    match self {
                        Self::Device(_, s) => s.index(index),
                        Self::Host(v) => v.index(index),
                    }
                }
            }

            impl<'a, T> IndexMut<$t> for HostOrDeviceSlice<'a, T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self::Output {
                    match self {
                        Self::Device(_, s) => s.index_mut(index),
                        Self::Host(v) => v.index_mut(index),
                    }
                }
            }
        )*
    }
}
impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<'a, T> Drop for HostOrDeviceSlice<'a, T> {
    fn drop(&mut self) {
        match self {
            Self::Device(device_id, s) => {
                if s.is_empty() {
                    return;
                }
                // free the cuda memory
                unsafe {
                    let _ = cudaSetDevice(*device_id);
                    cudaFree(s.as_mut_ptr() as *mut c_void).wrap().unwrap();
                }
            }
            Self::Host(_) => {}
        }
    }
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = usize; // This is a placeholder, TODO: actually make this into a proper CUDA wrapper
