import numpy as np
from importlib import import_module
from types import SimpleNamespace
import sys
sys.path.insert(0, "/home/smslab1/PycharmProjects/AnomalyDetection/Eagle/Eagle/src")
import patchcore.common

class Framework:
    """
    Orchestrates ExpertModel_PatchCore, Connector and MLLM.

    Usage:
      fw = Framework(expert, connector, mllm, args=args, device=device)
      processed_output, image_info = fw.evaluate(train_dataloader, patchcore_list,
                                                 test_image, args, image_path, model_name)
      # If you want to restore original AttentionPatcher after MLLM generation:
      fw.restore_attention_patch()
    """

    def __init__(self, expert, connector, mllm, args=None, device=None):
        self.expert = expert
        self.connector = connector
        self.mllm = mllm
        self.args = args
        self.device = device

        self.threshold = None
        self.score_max = None
        self.training_scores = None
        self.all_training_scores = {}
        self.all_labels_by_class = {}
        self.last_object_name = None

        # for monkey-patching mllm_base.AttentionPatcher
        try:
            self._mllm_base = import_module(".mllm_base", package=__package__)
        except Exception:
            import mllm_base as _mllm_base
            self._mllm_base = _mllm_base

        self._original_attention_patcher = getattr(self._mllm_base, "AttentionPatcher", None)
        self._patched_to_dummy = False

    def _train_and_compute_threshold(self, train_dataloader, patchcore_list, model_dir=None, save_artifacts=False, anomaly_labels=None):
        """
        Common helper:
         - run self.expert.fit(...)
         - set self.training_scores and self.score_max
         - compute threshold via self.connector.threshold_selection(...)
         - optionally save scores, models and threshold to model_dir if save_artifacts=True
        Returns: computed_threshold (float)
        """
        import logging
        LOGGER = logging.getLogger(__name__)
        LOGGER.debug("Running expert.fit to obtain training scores")

        aggregator = {"scores": []}
        scores = self.expert.fit(train_dataloader, patchcore_list)
        aggregator["scores"].append(scores)
        scores = np.array(aggregator["scores"])
        scores = np.mean(scores, axis=0)
        # scores = np.squeeze(np.array(scores))
        self.training_scores = scores
        self.score_max = float(np.max(scores)) if scores.size else None

        # compute threshold (use connector's selection logic)
        try:
            computed_threshold = float(self.connector.threshold_selection(scores))
        except Exception:
            # fallback: percentile using args.q or 99
            q = getattr(self.args, "q", 99)
            computed_threshold = float(np.percentile(np.ravel(scores), q))
        import os
        if save_artifacts and model_dir:
            os.makedirs(model_dir, exist_ok=True)
            # save scores
            np.save(os.path.join(model_dir, "scores.npy"), scores)
            # save models (ensemble naming)
            for i, PatchCore in enumerate(patchcore_list):
                prepend = f"Ensemble-{i + 1}-{len(patchcore_list)}_" if len(patchcore_list) > 1 else ""
                try:
                    PatchCore.save_to_path(model_dir, prepend)
                except Exception as e:
                    LOGGER.warning("Failed to save PatchCore[%d]: %s", i, str(e))
            # save threshold json
            import json
            try:
                with open(os.path.join(model_dir, "threshold.json"), "w") as f:
                    json.dump({"threshold": float(computed_threshold)}, f)
                with open(os.path.join(model_dir, "score_max.json"), "w") as f:
                    json.dump({"score_max": float(self.score_max)}, f)
            except Exception as e:
                LOGGER.warning("Failed to write threshold.json: %s", str(e))

        return computed_threshold, self.score_max


    def ensure_patchcore_for_object(
        self,
        object_name,
        train_dataloader,
        PatchCore_list,
        args,
        anomaly_labels,
    ):
        """
        Ensure PatchCore model and threshold exist for `object_name`.
        If saved artifacts exist -> load threshold and models.
        Otherwise -> train via helper and persist artifacts.
        Returns: optimal_image_threshold (float or None)
        """
        import logging
        import os
        import json
        device="cuda"
        LOGGER = logging.getLogger(__name__)
        model_dir = os.path.join("patchcore_models_paras", args.dataset_name, object_name)
        os.makedirs(model_dir, exist_ok=True)
        threshold_path = os.path.join(model_dir, "threshold.json")
        score_max_path = os.path.join(model_dir, "score_max.json")
        def _has_saved_patchcore(path):
            if not os.path.isdir(path):
                return False
            for fname in os.listdir(path):
                if fname.endswith(".faiss") or fname.endswith(".npy") or fname.endswith(".pth"):
                    return True
            return False

        saved_exists = _has_saved_patchcore(model_dir)
        optimal_image_threshold = None

        if saved_exists:
            LOGGER.info("Found saved PatchCore for %s. Loading from %s", object_name, model_dir)
            # try load threshold
            try:
                with open(threshold_path, "r") as f:
                    j = json.load(f)
                    optimal_image_threshold = float(j.get("threshold", j.get("threhold", 0.0)))
                LOGGER.info("Loaded threshold for %s: %f", object_name, optimal_image_threshold)

                with open(score_max_path, "r") as f:
                    j = json.load(f)
                    score_max = float(j.get("score_max", j.get("score_max", 0.0)))
                LOGGER.info("score_max for %s: %f", object_name, score_max)
            except FileNotFoundError:
                LOGGER.warning("No threshold.json found for %s, will need to re-estimate.", object_name)
                optimal_image_threshold = None
                score_max =None

            for i, PatchCore in enumerate(PatchCore_list):
                nn_method = patchcore.common.FaissNN(
                    getattr(args, "faiss_on_gpu", False),
                    getattr(args, "faiss_num_workers", 8),
                )
                prepend = f"Ensemble-{i + 1}-{len(PatchCore_list)}_" if len(PatchCore_list) > 1 else ""
                PatchCore.load_from_path(
                    load_path=model_dir,
                    device=device,
                    nn_method=nn_method,
                    prepend=prepend,
                )
        else:
            LOGGER.info("No saved PatchCore for %s. Training new model.", object_name)
            # Use shared helper and persist artifacts
            optimal_image_threshold,score_max = self._train_and_compute_threshold(
                train_dataloader,
                PatchCore_list,
                model_dir=model_dir,
                save_artifacts=True,
                anomaly_labels=anomaly_labels,
            )

            # bookkeeping for external use
            current_training_scores = np.squeeze(np.array(self.training_scores))
            current_labels = np.array(anomaly_labels)
            self.all_training_scores[object_name] = current_training_scores
            self.all_labels_by_class[object_name] = current_labels

        self.last_object_name = object_name
        # update instance threshold as well so evaluate() can use it
        if optimal_image_threshold is not None:
            self.threshold = optimal_image_threshold
        return optimal_image_threshold,score_max

    def predict(self, train_dataloader, test_image, patchcore_list, args):
        """
        Run expert.predict and reduce ensemble outputs to single score and anomaly map.
        """
        score, anomaly_map = self.expert.predict(train_dataloader, test_image, patchcore_list, args)
        score = np.array(score)
        score = np.mean(score, axis=0)
        # average across ensemble or dims similar to existing code
        if score.ndim > 1:
            score = np.mean(score, axis=0)
        anomaly_map = np.array(anomaly_map)
        if anomaly_map.ndim > 1:
            anomaly_map = np.mean(anomaly_map, axis=0)
        return score, anomaly_map

    def should_apply_attention(self, score):
        """
        Decide whether to apply attention editing:
          True if threshold < score < score_max
        """
        if self.threshold is None or self.score_max is None:
            return False
        try:
            s = float(np.mean(score)) if hasattr(score, "__len__") else float(score)
        except Exception:
            s = float(score)
        return (s > float(self.threshold)) and (s < float(self.score_max))

    def apply_attention_patch(self, enable: bool):
        """
        If enable==False, replace mllm_base.AttentionPatcher with a no-op dummy
        so downstream MLLM.generate_response won't modify attention.
        If enable==True, restore the original AttentionPatcher.
        """
        if enable:
            # restore original if previously patched
            if self._patched_to_dummy and self._original_attention_patcher is not None:
                setattr(self._mllm_base, "AttentionPatcher", self._original_attention_patcher)
                self._patched_to_dummy = False
        else:
            if not self._patched_to_dummy:
                class _DummyPatcher:
                    def __init__(self, cfg=None): pass
                    def enable(self): pass
                    def disable(self): pass
                    def set_cfg(self, cfg): pass
                setattr(self._mllm_base, "AttentionPatcher", _DummyPatcher)
                self._patched_to_dummy = True

    def restore_attention_patch(self):
        """
        Explicitly restore original AttentionPatcher (safe to call multiple times).
        """
        if self._original_attention_patcher is not None:
            setattr(self._mllm_base, "AttentionPatcher", self._original_attention_patcher)
            self._patched_to_dummy = False

    def evaluate(self, train_dataloader,object_name, patchcore_list, test_image, args, image_path, model_name, anomaly_labels=None, force_retrain=False):
        """
        End-to-end helper:
         - train/estimate threshold (if needed)
         - predict score & anomaly map
         - run connector.run_inference -> processed_output, image_info
         - decide and set AttentionPatcher state (but does NOT call MLLM.generate_response)
        Returns: (processed_output, image_info)
        Note: This function toggles the AttentionPatcher in `mllm_base` so that when
              the MLLM's generate_response is later called, attention editing is applied
              only when desired.
        """
        import logging
        LOGGER = logging.getLogger(__name__)
        # Only call ensure_patchcore_for_object when the object changed or when force_retrain requested.
        if force_retrain or (object_name != self.last_object_name):
            LOGGER.info("Object changed (last: %s -> current: %s) or force_retrain=%s. Ensuring PatchCore for object.",
                        self.last_object_name, object_name, force_retrain)
            self.threshold,self.score_max = self.ensure_patchcore_for_object(
                object_name=object_name,
                train_dataloader=train_dataloader,
                PatchCore_list=patchcore_list,
                args=args,
                anomaly_labels=anomaly_labels,
            )
        else:
            LOGGER.debug("Reusing cached PatchCore/threshold for object %s (last_object_name=%s).", object_name,
                         self.last_object_name)

        # predict on query
        score, anomaly_map = self.predict(train_dataloader, test_image, patchcore_list, args)

        # run connector inference to get processed_output and image_info
        processed_output, image_info = self.connector.run_inference(
            optimal_image_threshold=self.threshold,
            score_Test=score,
            anomaly_map_Test=anomaly_map,
            image_path=image_path,
            anomaly_labels=anomaly_labels,
            model_name=model_name,
        )
        if args.status:
            # decide attention editing and apply patch (monkey-patch AttentionPatcher as needed)
            apply_attention = self.should_apply_attention(score)
            return processed_output, image_info, apply_attention
        else:
            return processed_output, image_info ,None