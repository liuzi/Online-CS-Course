### 1. Compare all 4 settings (share or not * different loss weights)
* MSE increases through epochs during training but decreases during evaluation, indicating _**the regression model is overfit**_. Weighting more on MF model and totally sharing parameters can mitigate this problem, presented by the black line, grows much less than other three settings. Other solutions: implementing regularization techniques, simplifying the model, increasing training data, and using early stopping
![compare all on eval results](images/all_eval.png)
![compare all on training results](images/all_train.png)

### 2. Compare shared embeddings with different loss weights
* When totally share embeddings(parameters), half-half weighted loss damages the training on the first task: using matrix factorization to predict the probability p that a user would watch the movie.
![shared_eval](images/shared_eval.png)
![shared_train](images/shared_train.png)

### 3. Compare seperate embeddings with different loss weights
* If not share parameters at all, both weight settings meet overfit problem in regression model, while the performance on probability predicting task seems quite similar. 
![seperate_eval](images/seperate_eval.png)
* Weighting more on LF loss can reduce the joint loss during training process
![seperate_train](images/seperate_train.png)

### 4. Compare shared and seperate embeddings with loss weights = [0.5, 0.5]
* Sharing parameters can worsen the performance on probability prediction task (matrix factorization), while not affecting much on regression task.
![half_eval](images/half_eval.png)
* Sharing paramters increases loss on factorization task on joint loss.
![half_train](images/half_train.png)

### 5. Compare shared and seperate embeddings with loss weights = [0.99, 0.01]
* Increasingly weighting 99% on LF can improve factorization performance on validation set while sharing parameters
![mf_eval](images/mf_eval.png)
* Increasingly weighting 99% on LF negatively affect the training of regression task when sharing parameters.
![mf_train](images/mf_train.png)







