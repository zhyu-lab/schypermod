import math
import random
import numpy as np
import torch


class MaskSampler:
    def __init__(self, num_genes, modules, initial_mask_frac=0.15, final_mask_frac=0.35,
                 module_frac_start=0.1, module_frac_end=0.2, module_mask_frac=0.5,
                 curriculum_epochs=20, seed=3333):
        self.G = num_genes

        if isinstance(modules, dict):
            try:
                self.modules = [modules[k] for k in sorted(modules.keys())]
            except Exception:
                self.modules = list(modules.values())
        elif isinstance(modules, list):
            self.modules = modules
        else:
            raise ValueError(f"Unsupported modules format: {type(modules)}. Expected dict or list.")

        self.modules = [m for m in self.modules if hasattr(m, '__len__') and len(m) > 0]

        self.initial_mask_frac = initial_mask_frac
        self.final_mask_frac = final_mask_frac
        self.module_frac_start = module_frac_start
        self.module_frac_end = module_frac_end
        self.module_mask_frac = module_mask_frac
        self.curriculum_epochs = curriculum_epochs

        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.mask_frac = initial_mask_frac
        self.module_frac = module_frac_start

    def update_schedule(self, epoch):
        if epoch <= self.curriculum_epochs:
            t = (epoch - 1) / max(1, self.curriculum_epochs - 1)
            self.mask_frac = self.initial_mask_frac + t * (self.final_mask_frac - self.initial_mask_frac)
            self.module_frac = self.module_frac_start + t * (self.module_frac_end - self.module_frac_start)
        else:
            self.mask_frac = self.final_mask_frac
            self.module_frac = self.module_frac_end

    def _generate_mask(self, batch_size, target_mask_frac, target_module_frac):
        mask = torch.zeros(batch_size, self.G, dtype=torch.bool)

        num_mask = math.ceil(target_mask_frac * self.G)
        num_module_mask = math.ceil(num_mask * target_module_frac)
        num_random_mask = num_mask - num_module_mask

        for i in range(batch_size):
            modules_shuffled = list(self.modules)
            self.py_rng.shuffle(modules_shuffled)

            selected_module_genes = []
            total_mod = 0

            if num_module_mask > 0:
                for module in modules_shuffled:
                    if total_mod >= num_module_mask:
                        break
                    arr = np.array(module, dtype=np.int32)
                    if len(arr) == 0:
                        continue

                    n_sel = min(len(arr), max(1, int(len(arr) * self.module_mask_frac)), num_module_mask - total_mod)
                    if n_sel > 0:
                        sel = self.np_rng.choice(arr, size=n_sel, replace=False)
                        selected_module_genes.append(sel)
                        total_mod += n_sel

            mod_genes = np.concatenate(selected_module_genes) if selected_module_genes else np.array([], dtype=np.int32)
            mod_genes = mod_genes[:num_module_mask]

            mask_ind = np.zeros(self.G, dtype=bool)
            if len(mod_genes) > 0:
                mask_ind[mod_genes] = True

            all_genes = np.arange(self.G)
            avail = all_genes[~mask_ind]

            rand_genes = np.array([], dtype=np.int32)
            if num_random_mask > 0 and len(avail) > 0:
                act = min(num_random_mask, len(avail))
                rand_genes = self.np_rng.choice(avail, size=act, replace=False)

            final_genes = np.concatenate([mod_genes, rand_genes])
            if len(final_genes) > 0:
                mask[i, final_genes] = True

        return mask

    def sample_paired_views(self, batch_size):
        mask1 = self._generate_mask(batch_size, self.mask_frac, self.module_frac)
        mask2 = self._generate_mask(batch_size, self.mask_frac, self.module_frac)
        return mask1, mask2