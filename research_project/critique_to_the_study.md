Points to evaluate and that can go in the story line of the article:
- Actionable digital twins requires models that enable fault classification and characterisation
- Virtual sensors
- Tunable simulators that match the dynamics as they change over time
 
Questions/comments on the current results:
- Is the noise level realistic? Sounds low, but may be correct. Also, I would assume that the signal-to-noise level realistically would vary across channels? However, keeping it constant across channels makes it easier to disentangle various effects. Probably a 1% noise may be better. 
 
- Can parameter inference be improved by classification before estimation? In the first paper we claimed that the advantage of SBI was that we could do classification independently of parameter inference and thresholding. Hence if we do classification first, we can significantly reduce the inference complexity per failure but will have to train a posterior estimator for each failure mode.
 
- Notebook 24/25: Based on the SBI documentation, we should use 'zuko_nsf' insted of model='nsf'. https://sbi.readthedocs.io/en/latest/how_to_guide/03_density_estimators.html The SBI may have to be run again.
 
New with sbi  version 0.23: Note that "maf" or "nsf" correspond to nflows density estimators. Those have proven to work well, but the nflows package is not maintained anymore. To use more recent and actively maintained density estimators, we tentatively recommend using zuko, e.g., by passing zuko_maf or zuko_nsf.
 
density_estimator_build_fun = posterior_nn(
    model="zuko_nsf", hidden_features=60, num_transforms=3
)
inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
 
- Notebook 24/25: I have no feel for the complexity of the estimator neural net, but I'm guessing it can be smaller to speed up and to reduce risk of overfitting. Although the notebook claims no overfitting. This net is probably way too large in size (maybe depth is ok).
 
Potential areas for future research:
- Virtual sensor for x_D based on SBI? In the S-B scenario we can use SBI to recover x_D and use the information in the controller to break the degeracy?
- Model mis-specification and selection
      - This is partially done by others, but may still be relevant. In particular when the system is complicated and the model is known to have simplifying assumptions.
- Online training?
      - Initial training data only contains a couple of well known failure modes. New failure modes may occur over time. The pipeline should identify deviation from training distributions and update SBI using sequential SBI.
- Summary statistics/embeddings
      - Encoder/decoder
      - Time series data are correlated over time, should we rather use LSTM style encoder? Or SINDY style for explainability? Or simply just t-SNE