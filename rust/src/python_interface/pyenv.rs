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

use std::borrow::Borrow;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use crate::rl::env::Env;
use crate::python_interface::env::PyBaseEnv;


/// Try multiple strategies to clone a Python object.
/// Returns the cloned object or an error message describing what was tried.
fn try_clone_py_object(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Strategy 1: Try calling clone() method (custom clone protocol)
    if let Ok(cloned) = obj.call_method0(py, "clone") {
        return Ok(cloned);
    }

    // Strategy 2: Try calling copy() method
    if let Ok(copied) = obj.call_method0(py, "copy") {
        return Ok(copied);
    }

    // Strategy 3: Try calling __copy__() method
    if let Ok(copied) = obj.call_method0(py, "__copy__") {
        return Ok(copied);
    }

    // Strategy 4: Try copy.deepcopy (works if object implements pickle protocol)
    if let Ok(copy_module) = py.import("copy") {
        if let Ok(copied) = copy_module.call_method1("deepcopy", (obj,)) {
            return Ok(copied.unbind());
        }
    }

    // Strategy 5: Try pickle round-trip
    if let Ok(pickle_module) = py.import("pickle") {
        if let Ok(dumped) = pickle_module.call_method1("dumps", (obj,)) {
            if let Ok(loaded) = pickle_module.call_method1("loads", (dumped,)) {
                return Ok(loaded.unbind());
            }
        }
    }

    // All strategies failed
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Cannot clone environment. External environments must implement one of: \
         clone(), copy(), __copy__(), or be picklable. \
         Consider adding a clone() method to your environment class."
    ))
}


/// Wrapper for environments that implement the PyBaseEnv interface.
/// This is used for external Rust environments (like GridWorld) that extend PyBaseEnv
/// but are compiled in separate modules.
///
/// Method mapping:
/// - step(action) -> step(action)
/// - reward() -> reward()
/// - reset() -> reset() (no difficulty arg, uses setter)
pub struct PyBaseEnvWrapper {
    py_env: Py<PyAny>,
    difficulty: usize,
}

impl Clone for PyBaseEnvWrapper {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let copied = try_clone_py_object(py, &self.py_env)
                .expect("Failed to clone PyBaseEnv-like environment. See error above for details.");
            PyBaseEnvWrapper {
                py_env: copied,
                difficulty: self.difficulty,
            }
        })
    }
}

impl PyBaseEnvWrapper {
    pub fn new(py_env: Py<PyAny>) -> Self {
        PyBaseEnvWrapper { py_env, difficulty: 1 }
    }
}

impl Env for PyBaseEnvWrapper {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            // PyBaseEnv exposes difficulty as a property setter
            py_env.setattr(py, "difficulty", difficulty)
                .expect("Failed to set difficulty on PyBaseEnv-like environment");
        });
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env.call_method1(py, "set_state", (state,))
                .expect("Python `set_state` method failed.");
        });
    }

    fn num_actions(&self) -> usize {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "num_actions")
                .and_then(|val| val.extract::<usize>(py))
                .expect("Python `num_actions` method must return an integer.")
        })
    }

    fn obs_shape(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "obs_shape")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `obs_shape` method must return a list of integers.")
        })
    }

    fn reset(&mut self) {
        // Set difficulty first via the property, then call reset()
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env.setattr(py, "difficulty", self.difficulty)
                .expect("Failed to set difficulty");
            py_env
                .call_method0(py, "reset")
                .expect("Python `reset` method failed.");
        });
    }

    fn step(&mut self, action: usize) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            // PyBaseEnv uses step(action), not next(action)
            py_env
                .call_method1(py, "step", (action,))
                .expect("Python `step` method failed.");
        });
    }

    fn masks(&self) -> Vec<bool> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "masks")
                .and_then(|val| val.extract::<Vec<bool>>(py))
                .expect("Python `masks` method must return a list of booleans.")
        })
    }

    fn is_final(&self) -> bool {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "is_final")
                .and_then(|val| val.extract::<bool>(py))
                .expect("Python `is_final` method must return a boolean.")
        })
    }

    fn reward(&self) -> f32 {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            // PyBaseEnv uses reward(), not value()
            py_env
                .call_method0(py, "reward")
                .and_then(|val| val.extract::<f32>(py))
                .expect("Python `reward` method must return a float.")
        })
    }

    fn success(&self) -> bool {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            // Try to call success() if it exists, otherwise return is_final() as a fallback
            match py_env.call_method0(py, "success") {
                Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                Err(_) => self.is_final() && self.reward() > 0.0, // Fallback heuristic
            }
        })
    }

    fn observe(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "observe")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `observe` method must return a list of integers.")
        })
    }
}


/// Wrapper for pure Python environments that implement a custom interface.
///
/// Method mapping (different from PyBaseEnv):
/// - step(action) -> next(action)
/// - reward() -> value()
/// - clone -> copy()
pub struct PyEnvImpl {
    py_env: Py<PyAny>, // Reference to the Python object implementing the environment
    difficulty: usize
}


impl Clone for PyEnvImpl {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            PyEnvImpl {
                py_env: self.py_env.call_method0(py, "copy").expect("Python `copy` method failed."),
                difficulty: self.difficulty
            }
        })
    }
}


impl PyEnvImpl {
    /// Constructor: Create a new `PythonEnv` from a Python object
    pub fn new(py_env: Py<PyAny>) -> Self {
        PyEnvImpl { py_env, difficulty:1 }
    }
}

impl Env for PyEnvImpl {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    // Sets the current difficulty
    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    // Returns current difficulty
    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env.call_method1(py, "set_state", (state,))
                .expect("Python `set_state` method failed.");
        });
    }

    fn num_actions(&self) -> usize {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "num_actions")
                .and_then(|val| val.extract::<usize>(py))
                .expect("Python `num_actions` method must return an integer.")
        })
    }

    fn obs_shape(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "obs_shape")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `obs_shape` method must return a list of integers.")
        })
    }
    
    fn reset(&mut self) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method1(py, "reset", (self.difficulty,))
                .expect("Python `reset` method failed.");
        });
    }

    fn step(&mut self, action: usize) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method1(py, "next", (action,))
                .expect("Python `next` method failed.");
        });
    }

    fn masks(&self) -> Vec<bool> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "masks")
                .and_then(|val| val.extract::<Vec<bool>>(py))
                .expect("Python `masks` method must return a list of booleans.")
        })
    }

    fn is_final(&self) -> bool {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "is_final")
                .and_then(|val| val.extract::<bool>(py))
                .expect("Python `is_final` method must return a boolean.")
        })
    }

    fn reward(&self) -> f32 {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "value")
                .and_then(|val| val.extract::<f32>(py))
                .expect("Python `value` method must return a float.")
        })
    }

    fn success(&self) -> bool {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "success")
                .and_then(|val| val.extract::<bool>(py))
                .expect("Python `success` method must return a bool.")
        })
    }

    fn observe(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "observe")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `observe` method must return a list of integers.")
        })
    }
}


#[pyclass(subclass, extends=PyBaseEnv)]
pub struct PyEnv {}

#[pymethods]
impl PyEnv {
    #[new]
    pub fn new(py_env: Py<PyAny>) -> (Self, PyBaseEnv) {
        let env_impl = PyEnvImpl::new(py_env);
        let env = Box::new(env_impl);
        (PyEnv {}, PyBaseEnv { env: env })
    }
}