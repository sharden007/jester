from surprise import Dataset, SVD
from surprise.model_selection import cross_validate

# Load the Jester dataset (download it if not already available).
data = Dataset.load_builtin('jester')

# Use the Singular Value Decomposition (SVD) algorithm for recommendations
algo = SVD()

# Perform 5-fold cross-validation and evaluate the performance
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
