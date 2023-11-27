#[repr(C)]
pub enum NTTInputOutputOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(C)]
pub enum NTTDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(C)]
pub enum NTTType {
    Standard = 0,
    Coset = 1,
}