import matplotlib.pyplot as plt

N = [5,10,15,20,25]
N_sex = [10,15,20,25]

XX = 50

s0 = [6.93*0.1, 84.05*0.1, 671*0.1, 984*0.1, 1000*0.1,1000*0.1, 100]
p0_g = [100, 50, 6,0]
p0_a = [0,0,0,0]

p0_t = [0,50,94,100]

# plt.plot(N_sex, p0_g,N_sex,p0_a,"g", N_sex,p0_t, "r")
# plt.xlabel('N')
# plt.ylabel('[%]')
#
# plt.legend(["gamma", "alpha", "all alive"])
# plt.title("Probability of species dying first $\mathcal{S}_1$")
# plt.show()


s4 = [3.82*0.1, 26.99*0.1,62.22*0.1, 199.8*0.1, 558.82*0.1, 772.666*0.1, 917.6*0.1]
p4_g = [38,33,37,22]
p4_b = [63,67,63,45]
p4_a = [0,0,0,0]
p4_t = [0,0,0,33]

# plt.plot(N_sex, p4_g,N_sex,p4_b,N_sex,p4_a,N_sex,p4_t)
# plt.xlabel('N')
# plt.ylabel('[%]')
#
# plt.legend(["gamma", "beta", "alpha", "all alive"])
# plt.title("Probability of species dying first $\mathcal{S}_4$")
# plt.show()


s2 = [3.16*0.1,22.07*0.1,60.87*0.1,192.98*0.1,396.61*0.1, 645.133*0.1, 751*0.1]
p2_g = [71,56,36,28]
p2_b = [27,37,58,52]
p2_a = [2,7,6,12]

p2_t = [0,0,0,8]


# plt.plot(N_sex, p2_g,N_sex,p2_b,N_sex,p2_a,N_sex,p2_t)
# plt.xlabel('N')
# plt.ylabel('[%]')
#
# plt.legend(["gamma", "beta", "alpha", "all alive"])
# plt.title("Probability of species dying first $\mathcal{S}_2$")
# plt.show()

s3 = [3.44*0.1,19.56*0.1,43.53*0.1,104.7*0.1,164.07*0.1, 314.26*0.1, 575.7*0.1]
p3_g = [5,2,0,0]
p3_b = [95,98,100,99]
p3_a = [0,0,0,0]

p3_t = [0,0,0,1]

# plt.plot(N_sex, p3_g,N_sex,p3_b,N_sex,p3_a,N_sex,p3_t)
# plt.xlabel('N')
# plt.ylabel('[%]')
#
# plt.legend(["gamma", "beta", "alpha", "all alive"])
# plt.title("Probability of species dying first $\mathcal{S}_3$")
# plt.show()


N_n = [5,10,15,20,25,30,35]

#N = [5,10,15,20]
#s0 = [6.93*0.1, 84.05*0.1, 671*0.1, 984*0.1]

logi = []

for i in range(len(s0)):
    s0[i] = s0[i]*10
    s2[i] = s2[i] * 10
    s3[i] = s3[i] * 10
    s4[i] = s4[i] * 10
    logi.append(0.04*N_n[i]**3)


print(len(N), len(s0))
print(len(N_n))
print(len(s2), len(s3), len(s4))
plt.figure()
plt.loglog(N_n, s0, "x-")
plt.loglog(N_n,s2, "x-", N_n,s3, "x-", N_n,s4, "x-", N_n, logi)
#
# plt.plot(N_n, s0, "x-")
# plt.plot(N_n,s2, "x-", N_n,s3, "x-", N_n,s4, "x-", N_n, logi)

plt.xlabel('M')
plt.ylabel('[%]')

plt.legend(["$\mathcal{S}_1$", "$\mathcal{S}_2$", "$\mathcal{S}_3$", "$\mathcal{S}_4$", "$c \cdot M^3$"])
#plt.title("Average lenght of time before first species faces extinction")
plt.show()