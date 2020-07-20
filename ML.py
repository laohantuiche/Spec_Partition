import joblib
import time
import numpy as np
from benchs import benchs_all,benchs_MBA_4
from program_mgr import program_mgr
from keras.models import load_model

columns=['CPU_Utilization', 'Frequency', 'IPC', 'Misses', 'LLC', 'MBL', 'Memory_Footprint', 'Virt_Memory', 'Res_Memory','Allocated_Cache','Weighted_Speedup']
avg_columns=['Frequency', 'IPC', 'Misses']
sum_columns=['CPU_Utilization', 'LLC', 'MBL', 'Memory_Footprint', 'Virt_Memory', 'Res_Memory']
fix_columns=['Allocated_Cache', 'Weighted_Speedup']


class Model_Need_Protection:
	#1 需要保护，0 不需要保护
	def __init__(self):
		self.classifier=joblib.load('protected_or_not.pkl')
		self.max_arr=[1.00000e+02,3.30000e+03,3.55000e+00,1.53301e+05,2.53440e+04,1.29329e+04,4.00000e-01,9.87764e+05,8.84104e+05,5.17500e+01]
		self.min_arr=[0.00000e+00,9.99993e+02,2.30000e-01,2.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,1.59200e+04,5.68000e+02,-3.37500e+01]

	def call(self,input_arr):
		for i in range(10):
			input_arr[i] = (input_arr[i]-self.min_arr[i])/(self.max_arr[i]-self.min_arr[i])
		input_arr=np.asmatrix(input_arr)
		output=self.classifier.predict(input_arr)
		return int(output[0])


class Model_ST_or_CS:
	# 1是stream,0是cache-sensitive
	def __init__(self):
		self.classifier = joblib.load('stream_or_cache_sensitive.pkl')
		self.max_arr=[1.0000000e+02,3.3000150e+03,3.5500000e+00,1.5525500e+05,2.5344000e+04,1.2932900e+04,6.0000000e-01,1.4122760e+06,1.3631488e+06,8.1000000e+01]
		self.min_arr=[0.00000e+00,9.99499e+02,2.30000e-01,2.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,-1.03500e+02]
	def call(self,input_arr):
		for i in range(10):
			input_arr[i] = (input_arr[i]-self.min_arr[i])/(self.max_arr[i]-self.min_arr[i])

		input_arr=np.asmatrix(input_arr)
		output=self.classifier.predict(input_arr)
		return int(output[0])


class Model_seq2seq:
	def __init__(self):
		self.model = load_model('seq2seq.h5')
		self.max_arr=[[400.0, 3300.0, 2.4, 96091.0, 45792.0, 16603.100000000002, 1.4999999999999998, 3395028.0, 3321360.8, 14.36781547757468, 500.0, 3299.9995, 3.34, 90847.66666666669, 13320.0, 20827.800000000007, 1.0, 2459944.0, 2207220.0, 14.36781547757468],
						[700.0, 3300.0, 2.4, 88409.33333333331, 67824.0, 24244.9, 1.8, 4053204.0, 3985764.0, 14.905990020329845, 800.0, 3300.0, 3.31, 92144.66666666669, 15696.0, 21509.4, 1.4, 3268364.0, 3050464.0, 14.905990020329845],
						[400.0, 3300.0, 2.43, 91055.66666666669, 38016.0, 20681.3, 1.4999999999999998, 3360468.0, 3323284.8, 14.08217990946252, 500.0, 3300.0, 3.3600000000000003, 91186.0, 13032.0, 20739.5, 1.0, 2459948.0, 2205884.0, 14.08217990946252]]
		self.min_arr=[[87.5, 2502.096, 0.48, 135.0, 2376.0, 16.5, 0.1, 239600.0, 206328.0, 4.050699478954437, 93.8, 2619.7375, 0.87, 39.0, 0.0, 2.8, 0.0, 176048.0, 56836.0, 4.050699478954437],
						[87.5, 2501.8720000000008, 0.6066666666666666, 89.0, 1368.0, 7.0, 0.1, 241568.0, 230896.0, 4.170863292042881, 93.3, 1899.999, 0.87, 99.5, 0.0, 9.7, 0.0, 176048.0, 56864.0, 4.170863292042881],
						[87.5, 2502.37, 0.47, 100.0, 1584.0, 8.6, 0.2, 357764.0, 345752.0, 4.127057225429496, 93.3, 2620.5005, 0.8566666666666668, 51.0, 0.0, 2.7, 0.0, 176048.0, 56936.0, 4.127057225429496]]

	def call(self,input_arr):
		print(input_arr)
		for i in range(3):
			input_arr[i]=list(input_arr[i])
			for j in range(20):
				input_arr[i][j] = (input_arr[i][j]-self.min_arr[i][j])/(self.max_arr[i][j]-self.min_arr[i][j])
		input_arr=[input_arr]
		output=self.model.predict(input_arr)
		output=[arr[0] for arr in output[0]]
		return output

def main():
	#load model
	model_cs=Model_ST_or_CS()
	model_pro=Model_Need_Protection()
	model_s2s=Model_seq2seq()
	
	#run benchmarks
	benchmarks=benchs_MBA_4[0]
	mgr=program_mgr(benchmarks)
	mgr.start_all()
	time.sleep(5)

	#为了解决model_cs的问题，读入is_ST，根据文件判断程序是stream还是cache-sensitive
	with open('is_ST','r') as f:
		is_ST=eval(f.readline())

	#cluster stream and cache-sensitive
	for name,p in mgr.onfly().items():
		model_input=p.get_model_input()
		assert(model_input is not None)

		#model_output=model_cs.call(model_input)
		bench_id=p.bench_id if p.suite=='2017' else p.name
		model_output=1 if is_ST[bench_id] else 0

		print(name+' is a '+('stream' if model_output==1 else 'cache-sensitive')+' application')
		mgr.set_cluster(p,model_output)
	print(mgr.get_all_clusters())
	
	#collect 3 allocation cases
	arrs=[]
	for case in [[10,1],[9,2],[8,3]]:
		print('collecting trace of case:{}'.format(str(case)))
		LLC={0:case[0],1:case[1]}
		mgr.clus_allo_LLC(LLC)
		arr=mgr.get_cluster_s2s_input()	
		arrs.append(arr)
	model_input=arrs
	model_output=model_s2s.call(model_input)
	WSs=[model_input[i][9] for i in range(3)]
	WSs.extend(list(model_output))
	print(WSs)
	LLC_ST=WSs.index(max(WSs))+1
	LLC_CS=11-LLC_ST
	mgr.clus_allo_LLC({0:LLC_CS,1:LLC_ST})
	print('stream:{} ways, cache-sensitive:{} ways'.format(LLC_ST,LLC_CS))

	if LLC_ST > 1:
		#protect performance sensitive programs
		for name,p in mgr.get_p_in_cluster(1)['all'].items():
			model_input=p.get_model_input()
			model_output=model_pro.call(model_input)
			mgr.set_c_cluster(p,2)
			mgr.clus_allo_LLC({0:LLC_CS,1:LLC_ST-1,2:1})		
	
if __name__=='__main__':
	main()
