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

use pyo3::prelude::*;

use crate::rl::codec::{build_codec, ObservationCodec};
use crate::python_interface::error_mapping::MyError;

#[pyclass(name = "ObservationCodec")]
pub struct PyObservationCodec {
    codec: Arc<dyn ObservationCodec>,
}

impl Clone for PyObservationCodec {
    fn clone(&self) -> Self {
        Self {
            codec: Arc::clone(&self.codec),
        }
    }
}

#[pymethods]
impl PyObservationCodec {
    pub fn encode(&self, obs: Vec<Vec<usize>>) -> Vec<Vec<f32>> {
        self.codec.encode_indices(&obs)
    }

    #[getter]
    pub fn obs_size(&self) -> usize {
        self.codec.obs_size()
    }

    pub fn clone(&self) -> Self {
        self.clone()
    }
}

#[pyfunction]
pub fn make_observation_codec(
    kind: &str,
    num_slots: usize,
    domain_size: usize,
) -> PyResult<PyObservationCodec> {
    let codec = build_codec(kind, num_slots, domain_size).map_err(MyError::from)?;
    Ok(PyObservationCodec { codec })
}
