#[macro_use]
extern crate rustacuda;

use rustacuda::device::DeviceAttribute;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;
    
    // Get the first device
    let device = Device::get_device(0)?;

    
    let compute_major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let compute_minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    println!("gpu device compute_major: {:?}, compute_minor: {:?}", compute_major,compute_minor);
    Ok(())
}