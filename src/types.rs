
#[repr(C)]
#[derive(Debug, Clone)]
pub enum NTTInputOutputOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub enum NTTDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub enum NTTType {
    Standard = 0,
    Coset = 1,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct NTTConfig {
    pub batches: u32,
    /**< The number of NTTs to compute. Default value: 1. */
    pub order: NTTInputOutputOrder,
    /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
     *   `InputOutputOrder::NN`. */
    pub ntt_type: NTTType,
    pub extension_rate_bits: u32,
    pub are_inputs_on_device: bool, //**< True if inputs are on device and false if they're on host. Default value: false.
    pub are_outputs_on_device: bool, //**< If true, output is preserved on device, otherwise on host. Default value: false.
    pub with_coset: bool,
}

impl Default for NTTConfig {
    fn default() -> Self {
        Self {
            batches: 1,
            order: NTTInputOutputOrder::NN,
            ntt_type: NTTType::Standard,
            extension_rate_bits: 0,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            with_coset: false,
        }
    }
}
