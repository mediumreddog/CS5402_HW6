from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise import evaluate, print_perf
import os

file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

data.split(n_folds=3)

# algo = KNNBasic(sim_options = {'user_based': False})
# algo = KNNBasic(sim_options = {'user_based': True})
# algo = NMF()
# algo = SVD(biased=False) # PMF

#algo = SVD()
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

# algo = SVD(biased=False)  # PMF
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = NMF()
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': False})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

#algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

for i in range(15, 40):
    print "User Based - K: " + str(i)
    algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': True})
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)

for i in range(15, 40):
    print "Item Based - K: " + str(i)
    algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': False})
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)
