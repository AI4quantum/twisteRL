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

use pyo3::prelude::*;

// NN Module
use crate::python_interface::modules::PySequential;
use crate::python_interface::layers::{PyEmbeddingBag, PyLinear};
use crate::python_interface::policy::PyPolicy;
use crate::python_interface::codec::{PyObservationCodec, make_observation_codec};

fn init_nn_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbeddingBag>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyPolicy>()?;
    Ok(())
}

// Env Module
// use crate::envs::puzzle::Puzzle;
use crate::python_interface::pyenv::PyEnv;
use crate::python_interface::env::{PyPuzzleEnv, PyBaseEnv, solve_py, evaluate_py};

fn init_env_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPuzzleEnv>()?;
    m.add_class::<PyBaseEnv>()?;
    m.add_class::<PyEnv>()?;
    Ok(())
}

// Collector module
use crate::python_interface::collector::PyPPOCollector;
use crate::python_interface::collector::PyAZCollector;
use crate::python_interface::collector::PyCollectedData;

fn init_collector_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPPOCollector>()?;
    m.add_class::<PyAZCollector>()?;
    m.add_class::<PyCollectedData>()?;
    m.add_function(wrap_pyfunction!(solve_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_py, m)?)?;
    Ok(())
}

fn init_codec_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObservationCodec>()?;
    m.add_function(wrap_pyfunction!(make_observation_codec, m)?)?;
    Ok(())
}

fn add_twisterl_functionality(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_submod = PyModule::new(py, "nn")?;
    init_nn_module(&nn_submod)?;
    m.add_submodule(&nn_submod)?;

    let env_submod = PyModule::new(py, "env")?;
    init_env_module(&env_submod)?;
    m.add_submodule(&env_submod)?;

    let collector_submod = PyModule::new(py, "collector")?;
    init_collector_module(&collector_submod)?;
    m.add_submodule(&collector_submod)?;

    let codec_submod = PyModule::new(py, "codec")?;
    init_codec_module(&codec_submod)?;
    m.add_submodule(&codec_submod)?;

    Ok(())
}


// Full package
#[pymodule]
fn twisterl(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_twisterl_functionality(py, m)?;
    Ok(())
}
