import sys
import datetime
from datetime import date, timedelta, datetime

from pyspark.conf import SparkConf

import argparse
import databricks.koalas as ks

from pyspark.sql import SparkSession
from yeondys.BPModel import BPModel


def define_argparser():
	parser = argparse.ArgumentParser("Yeondys")

	parser.add_argument('--model', help='A model output path')
	parser.add_argument('--target-date', help='target date')
	parser.add_argument('--days', help='range days (target-date - days)')
	parser.add_argument('--brand-visit-cnt', help='input brand visit count')
	parser.add_argument('--user-visit-cnt', help='input user visit count')

	args = parser.parse_args()
	return (parser, args)


def get_source_data(date_list):
	sql = """ 
        with a as ( select cast(dt as string) as dt, cast(pcid as string) as userid, cast(itemid as string) as itemid, staytime from user_item_view where dt in {date_list}),
         b as (select cast(a.pcid as string) as cust_no from user_pcid_map a left join (select cast(cust_no as string) as cust_no, cast(ec_cust_grd_cd as string) as ec_cust_grd_cd from vip_user_info) b on a.userid = b.cust_no where b.ec_cust_grd_cd is not null),
         c as (select distinct cast(prd_id as string) as prd_cd, cast(brd_id as string) as brand_cd from product where deal_flag = 'N' ) 
         select a.dt, a.userid, c.brand_cd, a.staytime from a left join b on a.userid = b.cust_no left join c on a.itemid = c.prd_cd where b.cust_no is not null and c.prd_cd is not null
         """.format(date_list=date_list)

	# sql = """ select userid, brand_cd, staytime from members_kangcj.brand_preference_source limit 10000 """

	print("execute sql : {}".format(sql))

	return sql


def get_toddler_data():
	sql = """ select distinct brd_id from dealteminfo where cate1 in ('1378777', '1378789', '1378783', '1403692', '1378778', '1378790', '1378802')   """

	print("execute sql : {}".format(sql))
	return sql


def date_generator(begin_date, end_date):
	begin = datetime.strptime(begin_date, "%Y%m%d")
	end = datetime.strptime(end_date, "%Y%m%d")
	return [(begin + timedelta(days=x)).strftime("%Y%m%d") for x in range(0, (end - begin).days + 1)]


def main(args):
	print("main start1!")
	print(args)

	formatted_target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
	begin_date = formatted_target_date - timedelta(days=int(args.days))
	end_date = formatted_target_date

	date_list = date_generator(begin_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
	str_date_list = "('" + "','".join(date_list) + "')"
	print("date list : {}".format(str_date_list))

	model = BPModel()

	spark_session = SparkSession.builder \
		.appName('Yeondys') \
		.config("spark.driver.memory", "15g") \
		.enableHiveSupport() \
		.getOrCreate()

	corpus = ks.sql(get_source_data(str_date_list))
	if corpus:
		model.threshold_likes(corpus.to_pandas(), int(args.brand_visit_cnt), int(args.user_visit_cnt))
	# TEST용으로 진행 원래는 airflow 에 설정되어있는 5000, 5를 사용
	# model.threshold_likes(corpus.to_pandas(), 1, 1)
	else:
		print("data collect fail!")
		sys.exit()

	corpus = ks.sql(get_toddler_data())
	if corpus:
		model.get_toddler_brand(corpus.to_pandas())
	# TEST용으로 진행 원래는 airflow 에 설정되어있는 5000, 5를 사용
	# model.threshold_likes(corpus.to_pandas(), 1, 1)
	else:
		print("toddler_data collect fail!")
		sys.exit()

	print("start prepare")
	model.prepare(end_date.strftime('%Y%m%d'))

	model.process()

	# begin_date = formatted_target_date - timedelta(days=14)
	# end_date = formatted_target_date
	# corpus = ks.sql(get_order_data(begin_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))
	# print("koalas order data size : {}".format(corpus.size))
	# model.post_process(corpus.to_pandas())
	model.post_process()

	result_df = ks.from_pandas(model.final_df)
	result_df.to_table(name='members_kangcj.brand_preference', format='orc', mode='overwrite')


if __name__ == '__main__':
	parser, args = define_argparser()
	print("start job..")
	main(args)
# except Exception as ex:
#    print('error ', ex)