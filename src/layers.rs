use crate::config::ModelConfig;
use candle_core::{D, Error, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops::silu};

struct RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32)
    }
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(config: ModelConfig, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get((config.n_embed, config.n_mlp), "weight")?;
        let bias = vb.get((config.n_mlp,), "bias")?;
        let gate_proj = Linear::new(weight.clone(), Some(bias.clone()));
        let up_proj = Linear::new(weight.clone(), Some(bias.clone()));
        let down_proj = Linear::new(weight, Some(bias));
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
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
    fn new(config: ModelConfig) -> CandleResult<Self> {
        let d = config.n_embed as i64 / config.n_mlp as i64;
        let t = config.rope_theta;
        let inv_req_vec: Vec<_> = (0..d)
            .step_by(2)
            .map(|i| 1f32 / t.powf(i as f32 / d as f32))
            .collect();
        let inv_req_vec_len = inv_req_vec.len();
        let inv_req = Tensor::from_vec(inv_req_vec, (1, inv_req_vec_len), &config.device)?;
        Ok(Self { inv_req })
    }

    fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> CandleResult<(Tensor, Tensor)> {
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

struct CausalSelfAttention {
    n_heads: usize,
    n_kv_heads: usize,
    n_embed: usize,
    n_embed_per_head: usize,
    n_kv_embed: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl CausalSelfAttention {
    fn new(config: ModelConfig, vb: VarBuilder) -> CandleResult<Self> {
        let n_embed_per_head = config.n_embed / config.n_heads;
        let n_kv_embed = config.n_kv_heads * n_embed_per_head;

        let q_proj_weight = vb.get((config.n_embed, config.n_embed), "q_proj_weight")?;
        let q_proj_bias = vb.get((config.n_embed,), "q_proj_bias")?;

        let k_proj_weight = vb.get((config.n_embed, n_kv_embed), "k_proj_weight")?;
        let k_proj_bias = vb.get((n_kv_embed,), "k_proj_bias")?;

        let v_proj_weight = vb.get((config.n_embed, n_kv_embed), "v_proj_weight")?;
        let v_proj_bias = vb.get((n_kv_embed,), "v_proj_bias")?;

        let o_proj_weight = vb.get((config.n_embed, config.n_embed), "o_proj_weight")?;
        let o_proj_bias = vb.get((config.n_embed,), "o_proj_bias")?;

        Ok(Self {
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            n_embed: config.n_embed,
            n_embed_per_head,
            n_kv_embed,
            q_proj: Linear::new(q_proj_weight, Some(q_proj_bias)),
            k_proj: Linear::new(k_proj_weight, Some(k_proj_bias)),
            v_proj: Linear::new(v_proj_weight, Some(v_proj_bias)),
            o_proj: Linear::new(o_proj_weight, Some(o_proj_bias)),
        })
    }

    fn forward(&self, xs: &Tensor, cos: f32, sin: f32) -> CandleResult<Tensor> {
        let shape = xs.shape().into_dims();
        let [b, t, c] = shape.as_slice() else {
            return Err(Error::msg(format!("expected 3D tensor, got {:?}", shape)));
        };
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b.clone(), t.clone(), self.n_heads, self.n_embed_per_head))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b.clone(), t.clone(), self.n_kv_heads, self.n_embed_per_head))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b.clone(), t.clone(), self.n_kv_heads, self.n_embed_per_head))?
            .transpose(1, 2)?;
    }
}
