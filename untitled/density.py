import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from matplotlib import animation
import copy

# North = down, South = up, East = right, West = left - thats how i happened to write it.


class PPsystem:

    def __init__(self, init_cond, L, T, r=2, mod = True, g=100, b=100, a = 100):
        # initialize the AC-system

        self.mod = mod #if true - use the critter-modification

        if mod:
            self.b_g = 0.6 #(1-p) i.e. b_g = 0.4
            self.d_ga = 0.6
            self.d_gb = 0.2

            self.b_b = 0.4
            self.d_b = 0.3
            self.d_ba = 0.4

            self.d_vb = 0.9 #(1-p) i.e. d_vb=0.001

            self.b_a = 0.3
            self.d_a = 0.2
        else:
            self.b_g = 0.6  # (1-p) i.e. b_g = 0.4
            self.d_ga = 0.6
            self.d_gb = 0.4

            self.b_b = 0.7
            self.d_b = 0.1
            self.d_ba = 0.2

            self.d_vb = 0.999  # (1-p) i.e. d_vb=0.001

            self.b_a = 0.3
            self.d_a = 0.2

        # OG param

        #
        # self.b_g = 0.6 #(1-p) i.e. b_g = 0.4
        # self.d_ga = 0.6
        # self.d_gb = 0.2
        #
        # self.b_b = 0.4
        # self.d_b = 0.3
        # self.d_ba = 0.4
        #
        # self.d_vb = 0.999 #(1-p) i.e. d_vb=0.001
        #
        # self.b_a = 0.3
        # self.d_a = 0.2



        # constants
        self.L = L #size
        self.r = r #depth of moore neighborhood
        self.T = T #runtime


        #will change

        self.dt = [0]
        self.AC = copy.deepcopy(init_cond)  # initial matrix
        self.gamma = [g]
        self.beta = [b]
        self.alpha = [a]
        self.alive_g = g


        self.backup_move = [0] * self.L #naming is a relic from when i thought i would keep alternative choices for walking in case of crash - but decided against it
        for i in range(self.L):     #this matrix keeps track and makes sure those who didnt get to walk will still be copied over without having moved
            self.backup_move[i] = [0] * self.L


    def attack(self,x,y, cell):
        """Function that takes care of the attack phase of the reaction step"""
        #0=0, gamma=1, beta=2, alpha=3, beta_full = 21, alpha_full = 31

        if cell == 0:
            new_state = 0
        else:
            neighbors = [self.AC[(x+1)%self.L][y], self.AC[x-1][y], self.AC[x][y-1], self.AC[x][(y + 1)%self.L]]
            if cell == 1:
                P = ((1-self.d_ga)**neighbors.count(3))*((1-self.d_gb)**neighbors.count(2))
                r = random.random()
                if r < P:
                    new_state = 1
                else:
                    new_state = 0
                    self.alive_g -= 1 # we need to know how many gamma are alive at the reproduction phase
            elif cell == 3:
                P = ((1-self.d_ga)**neighbors.count(1))*((1-self.d_ba)**neighbors.count(2))
                r = random.random()
                if r < P:
                    new_state = 3
                else:
                    new_state = 31
            else:
                P = (1-self.d_ba)**neighbors.count(3)
                r = random.random()
                if r < P:
                    P2 = (1-self.d_gb)**neighbors.count(1)
                    r2 = random.random()
                    if r2 < P2:
                        if self.mod:
                            r3 = random.random()
                        else:
                            r3 = 1
                        P3 = (1-self.d_vb)**(self.beta[-1]/(self.L*self.L))
                        if r3 < P3:
                            new_state = 21
                        else:
                            new_state = 2
                    else:
                        new_state = 21
                else:
                    new_state = 0
        return new_state

    def reproduce(self, x,y, temp_AC):
        """Function that takes care of the reproduction phase of the reaction step.
        Will perform the step for one cell"""
        #0=0, gamma=1, beta=2, alpha=3, beta_full = 21, alpha_full = 31
        neighbors = [temp_AC[(x+1)%self.L][y], temp_AC[(x-1)%self.L][y], temp_AC[x][(y-1)%self.L], temp_AC[x][(y + 1) % self.L]]


        if neighbors.count(3)+neighbors.count(31) > 0: #only alpha can be born
            P = ((1-self.b_a)**neighbors.count(31))
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 3
        elif neighbors.count(2)+neighbors.count(21) > 0: #only beta can be born
            P = ((1-self.b_b)**neighbors.count(21))
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 2
        elif neighbors.count(1) > 0:
            P = (1-self.b_g)**(self.alive_g/(self.L*self.L))
            r = random.random()
            if r < P:
                new_state = 1
            else:
                new_state = 0
        else:
            new_state = 0

        return new_state

    def natural_death(self, cell):
        """Function that takes care of the natural deaths during the reproction step.
        Will perform the step for one cell"""
        if cell == 3 or cell == 31:
            P = self.d_a
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 3
        else:
            P = self.d_b
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 2


        return new_state


    def reaction(self):
        """Main function of the reaction step. Will loop over all cells twice. In the first loop,
        it will call attack() for all cells not in state s=0. In the second loop it will call reproduce() if s=0 and natural deaths if s=beta/alpha"""
        self.alive_g = self.gamma[-1]

        temp_AC = [0] * self.L
        for i in range(len(temp_AC)):
            temp_AC[i] = [0] * self.L

        #attack
        for x in range(self.L):
            for y in range(self.L):
                cell = self.AC[x][y]
                if cell != 0:
                    temp_AC[x][y] = self.attack(x,y, cell)

        #reproduce and die
        for x in range(self.L):
            for y in range(self.L):
                cell = temp_AC[x][y]
                if cell == 0:
                    self.AC[x][y] = self.reproduce(x,y,temp_AC)
                elif cell == 1:
                    self.AC[x][y] = 1
                else:
                    self.AC[x][y] = self.natural_death(cell)



    def get_quarters(self, x,y): #returns moore-neighbors for all 4 quarters
        """help function during movement phase, will return moore and von neumann neighbors for a cell"""
        n_NSEW = [[], [], [], []]

        for i in range(1, self.r + 1):
            for j in range(-i, i + 1):
                y_list = [j, j, i, -i]
                x_list = [i, -i, j, j]
                for k in range(4):
                    y_n = y + y_list[k]
                    x_n = x + x_list[k]
                    if y_n >= self.L: #modulo to loop around
                        y_n = y_n % self.L
                    if x_n >= self.L:
                        x_n = x_n % self.L
                    n_NSEW[k].append(self.AC[x_n][y_n])
        NSEW = [self.AC[(x+1)%self.L][y], self.AC[x - 1][y], self.AC[x][y-1], self.AC[x][(y + 1) % self.L]]
        return n_NSEW, NSEW


    def movement(self):
        """Main function for the movement phase"""

        temp_move = [] # in position (x,y) we will save those cells (i,j) that would like to walk here in the next step
        for i in range(self.L):
            temp_move.append([])
        for lst in temp_move:
            for i in range(self.L):
                lst.append([])

        for x in range(self.L):
            for y in range(self.L): #check every cell in AC
                cell = self.AC[x][y] #state of the cell
                if cell == 0:
                    continue
                else:
                    step_nsew = [[(x + 1) % self.L, y], [x - 1, y], [x, y - 1], [x, (y + 1) % self.L]]  # these are the possible places that cell can go
                    n_NSEW, NSEW = self.get_quarters(x, y)
                    comp = []
                    dir = []
                    if cell == 1: #gamma - find which of step_nsew are possible and out of those - which is the best?
                        for i in range(4):
                            if NSEW[i] == 0: #if the step is even possible
                                comp.append(n_NSEW[i].count(3)*self.d_ga + n_NSEW[i].count(2)*self.d_gb)
                                dir.append(step_nsew[i])

                        zipped_pairs = list(zip(comp, dir))
                        random.shuffle(zipped_pairs) # So we do not favor North direction over SEW etc.
                        if len(comp) == 0:
                            p_step = [x,y]
                        else:
                            val, p_step = min(zipped_pairs)

                        #pref_step = [k for _, k in sorted(zipped_pairs)] #all possible steps, sorted with preferred step first
                        #print(pr_step, pref_step[0])

                    elif cell == 3: #alpha - find which of step_nsew are possible and out of those - which is the best?
                        for i in range(4):
                            if NSEW[i] == 0: #if the step is even possible
                                comp.append(n_NSEW[i].count(1) * self.d_ga + n_NSEW[i].count(2) * self.d_ba)
                                dir.append(step_nsew[i])

                        zipped_pairs = list(zip(comp, dir))
                        random.shuffle(zipped_pairs) # So we do not favor North direction over SEW etc.
                        if len(comp) == 0:
                            p_step = [x,y]
                        else:
                            val, p_step = max(zipped_pairs)

                        #pref_step = [k for _, k in sorted(zipped_pairs)]  # all possible steps, sorted with preferred step first (if min)
                        #pref_step.reverse() #since we want maximum

                    else: #beta - find which of step_nsew are possible and out of those - which is the best?
                        for i in range(4):
                            if NSEW[i] == 0: #if the step is even possible
                                if self.mod:
                                    comp.append(n_NSEW[i].count(1)/max(0.9,n_NSEW[i].count(3)))
                                else:
                                    comp.append(n_NSEW[i].count(1))
                                dir.append(step_nsew[i])

                        zipped_pairs = list(zip(comp, dir))
                        random.shuffle(zipped_pairs) # So we do not favor North direction over SEW etc.
                        #pref_step = [k for _, k in sorted(zipped_pairs)]  # all possible steps, sorted with preferred step first (if min)
                        #pref_step.reverse() #since we want maximum

                        if len(comp) == 0:
                            p_step = [x,y]
                        else:
                            val, p_step = max(zipped_pairs)

                    #pref_step.append([x,y])
                    #p_step = pref_step[0]
                    self.backup_move[x][y] = 1
                    temp_move[p_step[0]][p_step[1]].append([x, y])


        #temp_AC = [0] * self.L
        #for i in range(len(temp_AC)):
        #    temp_AC[i] = [0] * self.L

        for i in range(self.L):
            for j in range(self.L):
                if len(temp_move[i][j]) == 0: #nobody wanted to step here
                    temp_move[i][j] = 0
                elif len(temp_move[i][j]) == 1: #only one wanted to step here, indicate that they are "done" by setting corresponding backup_move(i,j)=0
                    x = temp_move[i][j][0][0]
                    y = temp_move[i][j][0][1]
                    temp_move[i][j] = self.AC[x][y]
                    self.backup_move[x][y] = 0
                else: #more than one wanted to step here - randomly let one of them do so and set corresponding backup_move -> 0
                    suspects = temp_move[i][j]
                    R = random.randint(0,len(suspects)-1)
                    x = temp_move[i][j][R][0]
                    y = temp_move[i][j][R][1]
                    temp_move[i][j] = self.AC[x][y]
                    self.backup_move[x][y] = 0


        for i in range(self.L):
            for j in range(self.L):
                if self.backup_move[i][j] == 1: #this agent did not get to walk
                    temp_move[i][j] = self.AC[i][j] #stay in the same place
                    self.backup_move[i][j] = 0

        self.AC = temp_move



    def integrate(self):
        """integrator funtion, will perform all steps."""
        for t in range(self.T + 1):

            self.dt.append(t+1)
            self.reaction()
            self.movement()

            [gamma, beta, alpha] = [0,0,0]
            for lst in self.AC:
                gamma += lst.count(1)
                beta += lst.count(2)
                alpha += lst.count(3)
            self.gamma.append(gamma)
            self.beta.append(beta)
            self.alpha.append(alpha)





    def plot_population(self):

        plt.plot(self.dt, self.gamma,self.dt,self.beta,self.dt,self.alpha)
        plt.xlabel('t')
        plt.ylabel('Population')

        plt.legend(["gamma", "beta", "alpha"])
        plt.title("Population change over time")
        plt.show()


    def integrate_stable(self):
        """integrator funtion, will perform all steps."""
        for t in range(self.T + 1):

            self.dt.append(t+1)
            self.reaction()
            self.movement()

            [gamma, beta, alpha] = [0,0,0]
            for lst in self.AC:
                gamma += lst.count(1)
                beta += lst.count(2)
                alpha += lst.count(3)
            self.gamma.append(gamma)
            self.beta.append(beta)
            self.alpha.append(alpha)

            if gamma == 0 or alpha == 0 or beta == 0:
                print("time", t)
                print("gamma", gamma)
                print("beta", beta)
                print("alpha", alpha)
                break









def create_init(L = 10, g=0, b=0, a=0):
    init_cond = [0] * L
    for j in range(len(init_cond)):
        init_cond[j] = [0] * L

    i = 0
    while i < a:
        x = random.randint(0,L-1)
        y = random.randint(0, L - 1)
        if init_cond[x][y] == 0:
            init_cond[x][y] = 3
            i += 1
    i = 0
    while i < b:
        x = random.randint(0,L-1)
        y = random.randint(0, L - 1)
        if init_cond[x][y] == 0:
            init_cond[x][y] = 2
            i += 1
    i = 0
    while i < g:
        x = random.randint(0,L-1)
        y = random.randint(0, L - 1)
        if init_cond[x][y] == 0:
            init_cond[x][y] = 1
            i += 1

    return init_cond




def statistical_accuracy():

    M_list = [25,30,35,40,45,50]
    G_mean = []
    B_mean = []
    B_sem = []
    A_mean = []
    G_sem = []
    A_sem = []
    for M in M_list:
        size = M
        ini = round(size*size/16)
        time = 20
        eq = 100
        [L, T, g, b, a] = [size, time+eq, ini, ini, ini]
        runs = 10
        # sims = []
        # std_err_hist = [[],[]] #[[],[],[]]
        g_sem = 0
        a_sem = 0
        fr_g = []
        fr_b = []
        fr_a = []
        #init_cond = create_init(L, g, b, a)
        for i in range(runs):
            init_cond = create_init(L, g, b, a)
            PP = PPsystem(init_cond, L, T,g=g, b=b, a=a)
            PP.integrate()
            fr_g.append(sum(PP.gamma[eq:-1])/((time+1)*size*size))
            fr_b.append(sum(PP.beta[eq:-1])/((time+1)*size*size))
            fr_a.append(sum(PP.alpha[eq:-1])/((time+1)*size*size))


            if len(fr_g) == runs:
                var = [sum([f ** 2 for f in fr_g]) / len(fr_g) - (sum(fr_g) / len(fr_g)) ** 2,
                       sum([f ** 2 for f in fr_b]) / len(fr_g) - (sum(fr_b) / len(fr_g)) ** 2,
                       sum([f ** 2 for f in fr_a]) / len(fr_g) - (sum(fr_a) / len(fr_g)) ** 2]
                g_sem = var[0]
                b_sem = var[1]
                a_sem = var[2]

                #std_err = []
                # for j in range(len(var)):
                #     std_err_hist[j].append(math.sqrt(var[j] / (len(fr_g) - 1)))

                #std_err_hist.append(max(std_err))
                # sims.append(i)
        G_sem.append(g_sem)
        B_sem.append(b_sem)
        A_sem.append(a_sem)
        G_mean.append(sum(fr_g)/runs)
        B_mean.append(sum(fr_b)/runs)
        A_mean.append(sum(fr_a)/runs)


    print(M_list)

    print(G_mean)
    print(G_sem)
    print(B_mean)
    print(B_sem)
    print(A_mean)
    print(A_sem)

    plt.figure()
    plt.errorbar(M_list, G_mean, yerr=G_sem, capsize=2)
    plt.errorbar(M_list, B_mean, yerr=B_sem, capsize=2)
    plt.errorbar(M_list, A_mean, yerr=A_sem, capsize=2)
    plt.legend(["Gamma", "Beta", "Alpha"])
    # plt.plot(sims, std_err_hist[0], sims, std_err_hist[1], sims, std_err_hist[2])
    # plt.legend(["Gamma, mean: " + str(round(sum(fr_g)/runs)), "Beta, mean: " + str(round(sum(fr_b)/runs)), "Alpha, mean: " + str(round(sum(fr_a)/runs))])
    plt.xlabel('M')
    plt.ylabel('Population density')

    #plt.legend(["gamma", "beta", "alpha"])
    #plt.title("Population change over time")
    plt.show()



statistical_accuracy()
