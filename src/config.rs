use candle_core::Device;

pub struct ModelConfig {
    pub n_embed: usize,
    pub n_mlp: usize,
    pub rope_theta: f32,
    pub device: Device,
}
