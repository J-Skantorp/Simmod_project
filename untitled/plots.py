import matplotlib.pyplot as plt
#
# x = [0, 1, 2, 3, 4]
# y = [1, 3, 2, 4, 3]
# sd = [0.5, 0.25, 0.5, 0.75, 0.25]
# plt.errorbar(x, y, yerr=sd, capsize=2)
#
# x = [0, 1, 2, 3, 4]
# y = [1, 4, 4, 4, 3]
# sd = [0.5, 0.25, 0.5, 0.75, 0.25]
#
# plt.errorbar(x, y, yerr=sd, capsize=2)
# plt.show()
#
# # print(G_mean)
# print(G_sem)
# print(A_mean)
# print(A_sem)
#
#
# [25, 30, 35, 40]
# [0, 0, 0, 0]
# [2.436942153656585e-05, 2.776377713077996e-05, 2.70151220491191e-05, 3.7249281162152204e-05]
# [0, 0, 0, 0]
# [2.6503935221175445e-06, 2.7150129712077975e-06, 5.3301676878475535e-06, 4.841939722846633e-06]


# /usr/local/bin/python3.7 /Users/johannaskantorp/Documents/GitHub/Simmod_project/untitled/density.py THIS
# [25, 30, 35, 40]
# [0, 0, 0, 0]
# [0.00013727371551608414, 8.131694775864728e-05, 5.0585128063593277e-05, 4.3251177196543256e-05]
# [0, 0, 0, 0]
# [1.1225876417761405e-05, 9.308560758432283e-06, 6.0593113550971855e-06, 6.365160335045084e-06]


# /usr/local/bin/python3.7 /Users/johannaskantorp/Documents/GitHub/Simmod_project/untitled/density.py NO THIS!!


"""av density 2 agent"""
# M_list = [25, 30, 35, 40]
# G_mean = [0.28478922693266834, 0.28416248268218336, 0.28281345615552955, 0.2844305174563591]
# G_sem = [8.650313965707923e-05, 5.808027499375201e-05, 5.0657709122756134e-05, 2.675692265693974e-05]
# A_mean = [0.2509563690773067, 0.24995671931282906, 0.24978785688839125, 0.2501314526184538]
# A_sem = [1.2111800759884161e-05, 8.437673984097727e-06, 7.189553480342614e-06, 5.580609107894152e-06]
#
# plt.figure()
# plt.errorbar(M_list, G_mean, yerr=G_sem, capsize=2)
# plt.errorbar(M_list, A_mean, yerr=A_sem, capsize=2)
# locs, labels = plt.xticks()            # Get locations and labels
# ##plt.xticks(ticks, [labels], **kwargs)  # Set locations and labels
# plt.xticks(M_list)
# plt.legend(["Gamma", "Alpha"])
# # plt.plot(sims, std_err_hist[0], sims, std_err_hist[1], sims, std_err_hist[2])
# # plt.legend(["Gamma, mean: " + str(round(sum(fr_g)/runs)), "Beta, mean: " + str(round(sum(fr_b)/runs)), "Alpha, mean: " + str(round(sum(fr_a)/runs))])
# plt.xlabel('M')
# plt.ylabel('Population density')
#
# # plt.legend(["gamma", "beta", "alpha"])
# # plt.title("Population change over time")
# plt.show()

##### 3 agent system another run
# [25, 30, 35, 40]
# [0.20890701246882792, 0.1988354114713217, 0.19853936587103668, 0.19535202618453865]
# [0.0032626829629782, 0.0017997262756565685, 0.0009954745294602912, 0.0006170659470994128]
# [0.10704821945137157, 0.11918980326960377, 0.11990289582167028, 0.12718753117206982]
# [0.006887728492886885, 0.0035315578373422366, 0.002322199113801904, 0.0010541298239205611]
# [0.20533881296758108, 0.20332773621501807, 0.20472838312382308, 0.20057362842892767]
# [0.0013051197616078328, 0.0006221692235007445, 0.00039527405294177004, 0.00017700055201770593]

"""av density 3 agent - beta sem = wrong"""
# M_list = [25, 30, 35, 40]
# G_mean = [0.21990822942643387, 0.20680559711831534, 0.19511258588223318, 0.19235417705735663]
# G_sem = [0.0033108863696718546, 0.0016650505447258976, 0.0009965284762031498, 0.0005743250233557137]
# B_mean = [0.08727972069825438, 0.1081809365475201, 0.12393632245915823, 0.12646530548628432]
# B_sem = [0.006887728492886885, 0.0035315578373422366, 0.002322199113801904, 0.0010541298239205611]
# A_mean = [0.21555559102244387, 0.20781335550013855, 0.2029079546032877, 0.20252365960099752]
# A_sem = [0.0011297191351890695, 0.0007011206294660344, 0.0003548608069001147, 0.00021600498808405783]
#
#
# # print(M_list)
# #
# # print(G_mean)
# # print(G_sem)
# # print(B_mean)
# # print(B_sem)
# # print(A_mean)
# # print(A_sem)
#
# plt.figure()
# plt.errorbar(M_list, G_mean, yerr=G_sem, capsize=2)
# plt.errorbar(M_list, B_mean, yerr=B_sem, capsize=2)
# plt.errorbar(M_list, A_mean, yerr=A_sem, capsize=2)
# plt.legend(["Gamma", "Beta", "Alpha"])
# locs, labels = plt.xticks()            # Get locations and labels
# plt.xticks(M_list)
# # plt.plot(sims, std_err_hist[0], sims, std_err_hist[1], sims, std_err_hist[2])
# # plt.legend(["Gamma, mean: " + str(round(sum(fr_g)/runs)), "Beta, mean: " + str(round(sum(fr_b)/runs)), "Alpha, mean: " + str(round(sum(fr_a)/runs))])
# plt.xlabel('M')
# plt.ylabel('Population density')
#
# # plt.legend(["gamma", "beta", "alpha"])
# # plt.title("Population change over time")
# plt.show()


"""plot av with colorbar"""

# av_time = [[],[],[],[]]
# makes_it = [[],[],[],[]]
# # /usr/local/bin/python3.7 /Users/johannaskantorp/Documents/GitHub/Simmod_project/untitled/.idea/validity.py
# av_time[0] = [8.3, 14.44, 25.8, 29.64, 59.86, 113.52, 194.42, 258.68, 300.74, 479.56, 692.18, 828.42, 943.08, 946.34, 992.8, 976.24, 980.42, 1000.0, 1000.0, 993.32, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
# makes_it[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.16, 0.44, 0.66, 0.86, 0.86, 0.98, 0.96, 0.98, 1.0, 1.0, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# av_time[1] = [5.14, 7.6, 9.52, 14.3, 19.58, 24.2, 26.8, 33.52, 40.88, 48.26, 70.66, 89.08, 102.42, 126.44, 167.02, 164.66, 233.22, 271.98, 291.34, 345.28, 383.44, 424.9, 497.8, 463.18, 627.26, 581.34, 653.32, 616.0, 688.88, 652.54]
# makes_it[1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.04, 0.08, 0.18, 0.12, 0.32, 0.2, 0.32, 0.32, 0.34, 0.3]
# av_time[2] = [4.72, 7.46, 10.98, 13.18, 17.86, 21.52, 27.24, 26.76, 41.0, 48.74, 48.68, 56.76, 60.84, 81.7, 83.98, 89.24, 97.7, 110.76, 163.16, 137.6, 162.04, 183.5, 251.62, 250.48, 294.02, 365.36, 377.58, 382.1, 455.72, 512.56]
# makes_it[2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08, 0.14, 0.26]
# av_time[3] = [4.8, 7.28, 10.08, 13.8, 17.68, 23.06, 39.82, 46.82, 48.72, 69.04, 73.18, 79.18, 79.56, 123.7, 156.1, 213.44, 299.06, 329.68, 466.28, 482.7, 609.0, 704.82, 701.2, 696.3, 760.0, 900.48, 872.38, 981.58, 960.22, 941.58]
# makes_it[3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.1, 0.18, 0.28, 0.42, 0.48, 0.56, 0.5, 0.62, 0.8, 0.8, 0.98, 0.92, 0.92]
# M_vec = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
#
#
# logi_av = [[],[],[],[]]
#
# logi = []
# for i in range(len(M_vec)):
#     M = M_vec[i]
#     logi.append(min(1000,0.04 * M ** 3))
#     for k in range(4):
#         logi_av.append(av_time[k][i])
#
#
# plt.figure()
# plt.loglog(M_vec,av_time[0], M_vec,av_time[1], M_vec,av_time[2],  M_vec, av_time[3],  M_vec, logi)
#
# # ax = plt.gca()
# # ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')
#
# plt.scatter(M_vec,av_time[0], c=makes_it[0], cmap='plasma')
# plt.scatter(M_vec,av_time[1], c=makes_it[1], cmap='plasma')
# plt.scatter(M_vec,av_time[2], c=makes_it[2], cmap='plasma')
# plt.scatter(M_vec,av_time[3], c=makes_it[3], cmap='plasma')
# plt.colorbar()
#
# plt.xlabel('M')
# plt.ylabel('[%]')
#
# plt.legend(["$\mathcal{S}_1$", "$\mathcal{S}_2$", "$\mathcal{S}_3$", "$\mathcal{S}_4$", "$c \cdot M^3$"])
# #plt.title("Average lenght of time before first species faces extinction")
# plt.show()


"""plot av with colorbar"""
av_time = [[],[],[],[]]
makes_it = [[],[],[],[]]
# /usr/local/bin/python3.7 /Users/johannaskantorp/Documents/GitHub/Simmod_project/untitled/.idea/validity.py
av_time[0] = [8.3, 14.44, 25.8, 29.64, 59.86, 113.52, 194.42, 258.68, 300.74, 479.56, 692.18, 828.42, 943.08, 946.34, 992.8, 976.24, 980.42, 1000.0, 1000.0, 993.32, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
makes_it[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.16, 0.44, 0.66, 0.86, 0.86, 0.98, 0.96, 0.98, 1.0, 1.0, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
av_time[1] = [5.14, 7.6, 9.52, 14.3, 19.58, 24.2, 26.8, 33.52, 40.88, 48.26, 70.66, 89.08, 102.42, 126.44, 167.02, 164.66, 233.22, 271.98, 291.34, 345.28, 383.44, 424.9, 497.8, 463.18, 627.26, 581.34, 653.32, 616.0, 688.88, 652.54]
makes_it[1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.04, 0.08, 0.18, 0.12, 0.32, 0.2, 0.32, 0.32, 0.34, 0.3]
av_time[2] = [4.72, 7.46, 10.98, 13.18, 17.86, 21.52, 27.24, 26.76, 41.0, 48.74, 48.68, 56.76, 60.84, 81.7, 83.98, 89.24, 97.7, 110.76, 163.16, 137.6, 162.04, 183.5, 251.62, 250.48, 294.02, 365.36, 377.58, 382.1, 455.72, 512.56]
makes_it[2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08, 0.14, 0.26]
av_time[3] = [4.8, 7.28, 10.08, 13.8, 17.68, 23.06, 39.82, 46.82, 48.72, 69.04, 73.18, 79.18, 79.56, 123.7, 156.1, 213.44, 299.06, 329.68, 466.28, 482.7, 609.0, 704.82, 701.2, 696.3, 760.0, 900.48, 872.38, 981.58, 960.22, 941.58]
makes_it[3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.1, 0.18, 0.28, 0.42, 0.48, 0.56, 0.5, 0.62, 0.8, 0.8, 0.98, 0.92, 0.92]
M_vec_ = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]


# logi_av = [[],[],[],[]]
#
# logi = []
# for i in range(len(M_vec)):
#     M = M_vec[i]
#     logi.append(min(1000,0.04 * M ** 3))
#     for k in range(4):
#         logi_av.append(av_time[k][i])

M_vec = []
logi = []
M_logi = []
for i in range(len(M_vec_)):
    M = M_vec_[i]
    M_vec.append(M*M)
    if 0.04*M**3 <1000:
        logi.append(min(1000,0.04 * M ** 3))
        M_logi.append(M*M)
    for k in range(4):
        makes_it[k][i] = makes_it[k][i]*100



plt.figure()
plt.plot(M_vec,av_time[0], M_vec,av_time[1], M_vec,av_time[2],  M_vec, av_time[3],  M_logi, logi, "k")

# ax = plt.gca()
# ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')

plt.scatter(M_vec,av_time[0], c=makes_it[0], cmap='plasma')
plt.scatter(M_vec,av_time[1], c=makes_it[1], cmap='plasma')
plt.scatter(M_vec,av_time[2], c=makes_it[2], cmap='plasma')
plt.scatter(M_vec,av_time[3], c=makes_it[3], cmap='plasma')
#ax = plt.colorbar()
#ax.set_label("[%]", rotation=180)

plt.xlabel('M x M')
plt.ylabel('Average time')

plt.legend(["$\mathcal{S}_1$", "$\mathcal{S}_2$", "$\mathcal{S}_3$", "$\mathcal{S}_4$", "$c \cdot M^3$"])
#plt.title("Average lenght of time before first species faces extinction")
plt.show()