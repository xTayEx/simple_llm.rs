use candle_core::Device;

pub struct ModelConfig {
    pub n_embed: usize,
    pub n_mlp: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub rope_theta: f32,
    pub device: Device,
}
