from stable_baselines3.common.callbacks import BaseCallback

class DecreaseOmegaCoef(BaseCallback):

    def __init__(self, n_steps: int, omega_coef: float,verbose: int = 1):
        super(DecreaseOmegaCoef, self).__init__(verbose)
        self.n_steps = n_steps
        self.omgea_coef = omega_coef
        self.changed = False

    def _on_step(self) -> bool:
        if (not self.changed) and self.n_calls > self.n_steps: 
            self.training_env.venv.venv.env_method("changeOmegaCoef", self.omgea_coef)
            self.changed = True

        return True