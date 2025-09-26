// -*- coding: utf-8 -*-
/*
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use std::sync::Arc;

/// Trait implemented by observation codecs that transform raw environment
/// representations into sparse indices and dense vectors.
pub trait ObservationCodec: Send + Sync {
    fn obs_size(&self) -> usize;
    fn encode_state(&self, state: &[usize]) -> Vec<usize>;
    fn encode_indices(&self, obs: &[Vec<usize>]) -> Vec<Vec<f32>>;
}

/// Multi-hot codec that expands categorical slots into multi-hot vectors.
pub struct MultiHotCodec {
    num_slots: usize,
    domain_size: usize,
    obs_size: usize,
}

impl MultiHotCodec {
    pub fn new(num_slots: usize, domain_size: usize) -> Self {
        Self {
            num_slots,
            domain_size,
            obs_size: num_slots * domain_size,
        }
    }
}

impl ObservationCodec for MultiHotCodec {
    fn obs_size(&self) -> usize {
        self.obs_size
    }

    fn encode_state(&self, state: &[usize]) -> Vec<usize> {
        state
            .iter()
            .enumerate()
            .map(|(slot, &value)| {
                let v = if value < self.domain_size {
                    value
                } else {
                    value % self.domain_size
                };
                slot * self.domain_size + v
            })
            .collect()
    }

    fn encode_indices(&self, obs: &[Vec<usize>]) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(obs.len());
        for sample in obs {
            let mut dense = vec![0.0f32; self.obs_size];
            for &idx in sample {
                if idx < self.obs_size {
                    dense[idx] = 1.0;
                }
            }
            result.push(dense);
        }
        result
    }
}

pub fn build_codec(
    kind: &str,
    num_slots: usize,
    domain_size: usize,
) -> anyhow::Result<Arc<dyn ObservationCodec>> {
    match kind {
        "multi_hot" => Ok(Arc::new(MultiHotCodec::new(num_slots, domain_size))),
        _ => Err(anyhow::anyhow!("Unknown observation codec type: {}", kind)),
    }
}
