- simulator_opinion_dynamics.py is used for simulating the data traces

- MSM_calibrator.py and MSM_simulator.py are used for MSM estimation. They make use of calibrator_blackit (a slight variation of blackit package)

- msm_first_moments is not used directly in the notebooks, but it contains other distance measure that could be used with the MSM 

- calibrator_blackit.py is obtained from to blackit (https://github.com/bancaditalia/black-it)

- initialize_model.py and opinion_dynamics_models_estimation.py are used for ML estimation

- repeat_function contain methods for efficiently repeating the experiments
