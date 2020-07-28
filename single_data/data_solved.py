import re
import pandas as pd

benchmark_names = [
    '500.perlbench_r', '502.gcc_r', '503.bwaves_r', '505.mcf_r', '507.cactuBSSN_r', '508.namd_r', '511.povray_r',
    '519.lbm_r', '520.omnetpp_r', '523.xalancbmk_r', '525.x264_r', '526.blender_r', '527.cam4_r', '531.deepsjeng_r',
    '538.imagick_r', '541.leela_r', '544.nab_r', '548.exchange2_r', '549.fotonik3d_r', '554.roms_r', '557.xz_r',
    'astar_base', 'bzip2_base', 'cactusADM_base', 'calculix_base', 'dealII_base', 'gcc_base', 'GemsFDTD_base',
    'gobmk_base', 'gromacs_base', 'h264ref_base', 'hmmer_base', 'lbm_base', 'leslie3d_base', 'libquantum_base',
    'mcf_base', 'namd_base', 'omnetpp_base', 'perlbench_base', 'povray_base', 'sjeng_base', 'soplex_base',
    'tonto_base', 'Xalan_base'
]

data = []
for benchmark in benchmark_names:
    benchmark_set = {}
    for cache_way in range(11):
        file = open(benchmark+'\\'+benchmark+'_'+str(cache_way+1), 'r')
        # print(file.name)
        file_data = file.readlines()
        sum_of_ipc = 0
        sum_of_cm = 0
        length = 0
        for i in range(len(file_data)):
            if i == 0:
                continue
            else:
                line_str = re.split(',', file_data[i][:-1])
                if line_str[3] != '' and line_str[4] != '':
                    instruction_per_clock = float(line_str[3])
                    cache_misses = float(line_str[4])
                    sum_of_ipc += instruction_per_clock
                    sum_of_cm += cache_misses
                    length += 1
                    if length > 30:
                        break
                else:
                    continue
        average_ipc = sum_of_ipc/(length)
        average_cm = sum_of_cm/(length)
        benchmark_set[cache_way+1] = [average_ipc, average_cm]
    data.append(benchmark_set)

sheet_file = open('curve.csv', 'w')
for i, benchmark in enumerate(benchmark_names):
    temp_str = benchmark + ','
    for j in range(1, 12):
        temp_str += str(data[i][j][0]) + ',' + str(data[i][j][1]) + ','
    sheet_file.write(temp_str[:-1]+'\n')
