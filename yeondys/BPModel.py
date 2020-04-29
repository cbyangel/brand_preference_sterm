import _pickle as cPickle
import gzip
import pandas as pd
import numpy as np
import pickle as pk

import os
import time

from datetime import date

from surprise import Reader
from surprise import Dataset
from surprise import NMF


class BPModel:

	def __init__(self, path=None, sc=None):
		self.prepare_dataframe = None
		self.final_df = None
		self.toddler_brand_df = None
		self.toddler_brand_list = None
		self.filter_cus_target = None
		self.vip_dataset_df = None
		self.nmf_result = {}
		self.model_param_list = [{'n_factors': 20, 'n_epochs': 30}]
		if path is not None:
			self.load(path, sc)

	def threshold_likes(self, df, uid_min, mid_min):
		n_users = df.userid.unique().shape[0]
		n_items = df.brand_cd.unique().shape[0]
		n_total = df.shape[0]
		sparsity = float(df.shape[0]) / float(n_users * n_items) * 100
		print('Starting likes info')
		print('Number of users: {}'.format(n_users))
		print('Number of brands: {}'.format(n_items))
		print('Number of total: {}'.format(n_total))
		print('Sparsity: {:4.3f}%'.format(sparsity))
		print("")

		done = False
		while not done:
			starting_shape = df.shape[0]
			mid_counts = df.groupby('userid').brand_cd.count()
			df = df[~df.userid.isin(mid_counts[mid_counts < mid_min].index.tolist())]
			uid_counts = df.groupby('brand_cd').userid.count()
			df = df[~df.brand_cd.isin(uid_counts[uid_counts < uid_min].index.tolist())]
			ending_shape = df.shape[0]
			if starting_shape == ending_shape:
				done = True

		n_users = df.userid.unique().shape[0]
		n_items = df.brand_cd.unique().shape[0]
		n_total = df.shape[0]
		sparsity = float(df.shape[0]) / float(n_users * n_items) * 100
		print('Ending likes info')
		print('Number of users: {}'.format(n_users))
		print('Number of brands: {}'.format(n_items))
		print('Number of total: {}'.format(n_total))
		print('Sparsity: {:4.3f}%'.format(sparsity))
		self.prepare_dataframe = df

	def get_toddler_brand(self, df):
		self.toddler_brand_df = df
		self.toddler_brand_df['brd_id'] = self.toddler_brand_df['brd_id'].astype(str)
		self.toddler_brand_list = self.toddler_brand_df['brd_id'].unique().tolist()
		print("test_toddler")

	def prepare(self, end_date):
		print('start prepare.')
		# 머무는 시간
		print("start day weight!!")
		self.prepare_dataframe['duration'] = (
					pd.to_datetime(end_date) - pd.to_datetime(self.prepare_dataframe['dt'])).dt.days
		self.prepare_dataframe['staytime'] = self.prepare_dataframe['staytime'].astype(float)
		# self.prepare_dataframe['staytime'] = np.round(self.prepare_dataframe['staytime'] * np.power(0.96, self.prepare_dataframe['duration'] + 1)).astype(int)
		self.prepare_dataframe = self.prepare_dataframe[['userid', 'brand_cd', 'staytime']]
		print("end day weight!!")
		print(self.prepare_dataframe.head())
		# self.prepare_dataframe['staytime'] = self.prepare_dataframe['staytime'].astype(int)
		self.prepare_dataframe.set_index(['userid', 'brand_cd'], inplace=True)

		vip_base_df = self.prepare_dataframe.groupby(['userid'])['staytime'].sum().reset_index(name='tot_stay')
		vip_stime_df = self.prepare_dataframe.groupby(['userid', 'brand_cd'])['staytime'].sum().reset_index(name='stay')
		vip_stime_df = vip_stime_df.merge(vip_base_df, on='userid', how='left')
		vip_stime_df['staytime_ratio'] = np.where(vip_stime_df['stay'] == 0, 0,
		                                          vip_stime_df['stay'] / vip_stime_df['tot_stay'])
		vip_stime_df = vip_stime_df[['userid', 'brand_cd', 'staytime_ratio']]
		vip_stime_df = vip_stime_df[vip_stime_df.staytime_ratio > 0]

		brand_filter_df = vip_stime_df.merge(self.toddler_brand_df, left_on='brand_cd', right_on='brd_id', how='left')
		cust_no_toddler = set(brand_filter_df.userid.unique()) - set(
			brand_filter_df[brand_filter_df.brd_id.notnull()]['userid'].unique())
		self.filter_cus_target = cust_no_toddler
		self.vip_dataset_df = vip_stime_df
		# cust_toddler = set(brand_filter_df.userid.unique()) - cust_no_toddler
		# vip_no_toddler_df = vip_stime_df.loc[vip_stime_df['userid'].isin(cust_no_toddler)]
		# vip_toddler_df = vip_stime_df.loc[vip_stime_df['userid'].isin(cust_toddler)]
		# self.vip_dataset_list.append(vip_stime_df)
		# self.vip_dataset_list.append(vip_no_toddler_df)
		# self.vip_dataset_list.append(vip_toddler_df)
		print('end prepare.')

	# end 학습

	def process(self):
		print('start process')

		# user_brand_list = []
		model_param = self.model_param_list[0]
		reader = Reader(line_format='user item rating', rating_scale=(0, 1))

		print("count vip_dataset_df : {}".format(len(self.vip_dataset_df)))
		# for i in range(0, len(self.vip_dataset_list)):
		# data = Dataset.load_from_df(df=self.vip_dataset_list[i][['userid', 'brand_cd', 'staytime_ratio']], reader=reader)
		data = Dataset.load_from_df(df=self.vip_dataset_df[['userid', 'brand_cd', 'staytime_ratio']], reader=reader)

		# 학습
		algo = NMF(
			n_factors=model_param['n_factors'],
			n_epochs=model_param['n_epochs']
		)

		train_data = data.build_full_trainset()
		algo.fit(train_data)
		nmf_model = algo

		model_trainset = nmf_model.trainset
		user_name_to_id_dict = model_trainset._raw2inner_id_users
		brand_name_to_id_dict = model_trainset._raw2inner_id_items

		user_name_list = list(user_name_to_id_dict.keys())
		brand_name_list = list(brand_name_to_id_dict.keys())
		print("user_name_list:", len(user_name_list))
		print("item_name_list:", len(brand_name_list))

		all_rating_list = list(model_trainset.all_ratings())

		user_brand_score_dict = {}
		for item in all_rating_list:
			inner_user = item[0]
			inner_brand = item[1]
			score = item[2]
			if score == 0:
				continue
			user_brand_score_dict.setdefault(inner_user, {})
			user_brand_score_dict[inner_user].setdefault(inner_brand, score)

		model_user_brand_score_dict = {}
		now = time.time()
		user_index = 0
		for user_name in user_name_list:
			if user_index % 50000 == 0:
				print(user_index)
			user_index += 1

			inner_user_id = user_name_to_id_dict[user_name]
			score_list = []

			for brand_name in brand_name_list:
				inner_brand_id = brand_name_to_id_dict[brand_name]

				if inner_brand_id not in user_brand_score_dict[inner_user_id]:
					## 없을 때에는 기존 uid와 bid로 넣어 예측값 가져옴
					score_list.append(nmf_model.predict(user_name, brand_name).est)
				else:
					## 있는 것은 그대로 가져옴.
					true_score = user_brand_score_dict[inner_user_id][inner_brand_id]
					score_list.append(true_score)

			score_list = [round(score, 3) for score in score_list]
			sort_index = np.argsort(score_list)[::-1][:10]
			model_user_brand_score_dict.setdefault(user_name, {})
			if user_name in self.filter_cus_target:
				model_user_brand_score_dict[user_name]['brand'] = [brand_name_list[index] for index in sort_index if
				                                                   brand_name_list[
					                                                   index] not in self.toddler_brand_list]
				model_user_brand_score_dict[user_name]['score'] = [score_list[index] for index in sort_index if
				                                                   brand_name_list[
					                                                   index] not in self.toddler_brand_list]
			else:
				model_user_brand_score_dict[user_name]['brand'] = [brand_name_list[index] for index in sort_index]
				model_user_brand_score_dict[user_name]['score'] = [score_list[index] for index in sort_index]

		self.nmf_result = model_user_brand_score_dict
		print('end process')

	def post_process(self):
		print('start post_process')
		userid_list = [k for k, v in self.nmf_result.items()]
		brand_list = [v['brand'] for k, v in self.nmf_result.items()]
		score_list = [v['score'] for k, v in self.nmf_result.items()]
		# prdid_list = [ord_list[k] if k in ord_list else '' for k, v in self.nmf_result.items()]

		print('userid_list count : {}'.format(len(userid_list)))
		print('brand_list count : {}'.format(len(brand_list)))
		print('score_list count : {}'.format(len(score_list)))
		# print('prdid_list count : {}'.format(len(prdid_list)))

		self.final_df = pd.DataFrame(list(zip(userid_list, brand_list, score_list)),
		                             columns=['userid', 'brand', 'score'])
		self.final_df['brand'] = [self.try_join(l) for l in self.final_df['brand']]
		self.final_df['score'] = [self.try_join(l) for l in self.final_df['score']]
		self.final_df['prdid'] = ''
		# self.final_df['prdid'] = [self.try_join(l) for l in self.final_df['prdid']]
		print(self.final_df.head())
		print("final_df count : {}".format(self.final_df.shape[0]))
		print('end post_process!')

	def try_join(self, l):
		try:
			return ','.join(map(str, l))
		except TypeError:
			return np.nan

