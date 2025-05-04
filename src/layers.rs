use candle_core::{Tensor, Result as candle_result, D};
use candle_nn::{ops::silu, Linear, Module, VarBuilder};
use crate::config::ModelConfig;

struct RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> candle_result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32) 
    } 
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(config: ModelConfig, vb: VarBuilder) -> candle_result<Self> {
        
        let weight = vb.get((config.n_embed, config.n_mlp), "weight")?;
        let bias = vb.get((config.n_mlp,), "bias")?;
        let gate_proj = Linear::new(weight.clone(), Some(bias.clone()));
        let up_proj = Linear::new(weight.clone(), Some(bias.clone()));
        let down_proj = Linear::new(weight, Some(bias));
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> candle_result<Tensor> {
        let gate_result = silu(&self.gate_proj.forward(xs)?)?;
        let up_result = self.up_proj.forward(xs)?;
        let mul_result = (gate_result * up_result)?;
        self.down_proj.forward(&mul_result)
    }
}

struct RotaryEmbedding {
    inv_req: Tensor,
}

impl RotaryEmbedding {
    fn new(config: ModelConfig) -> candle_result<Self> {
        let d = config.n_embed as i64 / config.n_mlp as i64;
        let t = config.rope_theta;
        let inv_req_vec : Vec<_> = (0..d).step_by(2).map(|i| 1f32 / t.powf(i as f32 / d as f32)).collect();
        let inv_req_vec_len = inv_req_vec.len();
        let inv_req = Tensor::from_vec(inv_req_vec, (1, inv_req_vec_len), &config.device)?;
        Ok(Self { inv_req })
    }

    fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> candle_result<(Tensor, Tensor)> {
        let pos_ids_dim = position_ids.dims().len();
        let inv_freq = if pos_ids_dim == 2 {
            &self.inv_req
        } else {
            &self.inv_req.unsqueeze(0)?.unsqueeze(0)?
        };
        
        let position_ids_unsqueezed = position_ids.unsqueeze(D::Minus1)?;
        let freq = (position_ids_unsqueezed * inv_freq)?;
        let emb = candle_core::Tensor::cat(&[&freq, &freq], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(xs.dtype())?;
        let sin = emb.sin()?.to_dtype(xs.dtype())?;

        // return cos and sin
        Ok((cos, sin))
    }
}

