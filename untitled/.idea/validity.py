import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
import numpy as np

from matplotlib import animation

# North = down, South = up, East = right, West = left - thats how i happened to write it.
import copy


class PPsystem:

    def __init__(self, init_cond, L, T, r=2, mod = True, intro = 0, T_intro = 20, n_intro = 20, animate=True, VB=0.9, two=False):
        # initialize the AC-system

        self.mod = mod #if true - use the critter-modification

        self.intro = intro # if 1/2/3 this species will be introduced later
        self.T_intro = T_intro #at time
        self.n_intro = n_intro #this many

        self.anim = animate #set to false if you do not intend to animate to save memory/time

        self.two = two

        if mod:
            self.b_g = 0.6 #(1-p) i.e. b_g = 0.4
            self.d_ga = 0.6
            self.d_gb = 0.2

            self.b_b = 0.4
            self.d_b = 0.3
            self.d_ba = 0.4

            self.d_vb = VB #(1-p) i.e. d_vb=0.001

            self.b_a = 0.3
            self.d_a = 0.2
        else:
            self.b_g = 0.6  # (1-p) i.e. b_g = 0.4
            self.d_ga = 0.6
            self.d_gb = 0.4

            self.b_b = 0.7
            self.d_b = 0.1
            self.d_ba = 0.2

            self.d_vb = 0.9  # (1-p) i.e. d_vb=0.001

            self.b_a = 0.3
            self.d_a = 0.2





        self.proc_b = 0.9 #not relevant
        self.proc_a = 0.9 #not relevant


        # constants
        self.L = L #size
        self.r = r #depth of moore neighborhood
        self.T = T #runtime


        #will change

        self.dt = [0]
        self.AC = copy.deepcopy(init_cond)  # initial matrix
        self.gamma = []
        self.beta = []
        self.alpha = []
        self.save_AC = [] #will save if animate = True

        gamma = 0
        beta = 0
        alpha = 0
        for lst in self.AC:
            gamma += lst.count(1)
            beta += lst.count(2)
            alpha += lst.count(3)
        self.gamma.append(gamma)
        self.beta.append(beta)
        self.alpha.append(alpha)
        self.alive_g = self.gamma[-1]



    def introduce(self):
        """Function to introduce a type of agent into the system at a later time
        will run at T_intro if intro=1,2 or 3"""
        s = self.intro
        a = self.n_intro
        i = 0
        j = 0
        while i < a:
            j += 1
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            if self.AC[x][y] == 0:
                self.AC[x][y] = s
                i += 1
            if j > self.L*self.L:
                i = a


    def attack(self,x,y):
        """Function that takes care of the attack phase of the reaction step"""
        #0=0, gamma=1, beta=2, alpha=3, beta_full = 21, alpha_full = 31

        cell = self.AC[x][y]
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
        if cell == 3:
            P = self.d_a
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 3
        elif cell == 31:
            P = self.d_a * self.proc_a
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 3
        elif cell == 2:
            P = self.d_b
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 2
        elif cell == 21:
            P = self.d_b * self.proc_b
            r = random.random()
            if r < P:
                new_state = 0
            else:
                new_state = 2
        else:
            new_state = 1
            print("something is wrong")

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
                temp_AC[x][y] = self.attack(x,y)

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
        backup_move = [0] * self.L #naming is a relic from when i thought i would keep alternative choices for walking in case of crash - but decided against it
        for i in range(self.L):     #this matrix keeps track and makes sure those who didnt get to walk will still be copied over without having moved
            backup_move[i] = [0] * self.L

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
                    backup_move[x][y] = 1
                    temp_move[p_step[0]][p_step[1]].append([x, y])


        temp_AC = [0] * self.L
        for i in range(len(temp_AC)):
            temp_AC[i] = [0] * self.L

        for i in range(self.L):
            for j in range(self.L):
                if len(temp_move[i][j]) == 0: #nobody wanted to step here
                    temp_AC[i][j] = 0
                elif len(temp_move[i][j]) == 1: #only one wanted to step here, indicate that they are "done" by setting corresponding backup_move(i,j)=0
                    x = temp_move[i][j][0][0]
                    y = temp_move[i][j][0][1]
                    temp_AC[i][j] = self.AC[x][y]
                    backup_move[x][y] = 0
                else: #more than one wanted to step here - randomly let one of them do so and set corresponding backup_move -> 0
                    suspects = temp_move[i][j]
                    R = random.randint(0,len(suspects)-1)
                    x = temp_move[i][j][R][0]
                    y = temp_move[i][j][R][1]
                    temp_AC[i][j] = self.AC[x][y]
                    backup_move[x][y] = 0


        for i in range(self.L):
            for j in range(self.L):
                if backup_move[i][j] == 1: #this agent did not get to walk
                    temp_AC[i][j] = self.AC[i][j] #stay in the same place
                    backup_move[i][j] = 0 #superflous but i like that backup_move is empty when we are done

        self.AC = temp_AC


    def integrate(self):
        """integrator funtion, will perform all steps."""
        for t in range(self.T + 1):
            if t == self.T_intro:
                if self.intro:
                    self.introduce()

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
            if self.anim:
                self.save_AC.append(self.AC)




    def animate(self):
        """Function to animate the AC, this function will"""
        for t in range(self.T + 1):
            plt.clf()

            cmap = (mpl.colors.ListedColormap(['white', 'green', 'blue', 'red'])
                    .with_extremes(over='0.25', under='0.75'))

            bounds = [0, 1, 2, 3,4]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

            plt.imshow(self.save_AC[t], cmap=cmap, interpolation='nearest')
            plt.colorbar()
            plt.title("t = " + str(t))

            plt.pause(0.1)


    def anim_and_population_plot(self):
        for t in range(self.T + 1):
            plt.clf()

            cmap = (mpl.colors.ListedColormap(['white', 'green', 'blue', 'red'])
                    .with_extremes(over='0.25', under='0.75'))

            bounds = [0, 1, 2, 3,4]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

            plt.subplot(2, 1, 2)
            plt.plot(self.dt[:t], self.gamma[:t], 'g')
            plt.plot(self.dt[:t], self.beta[:t], 'b')
            plt.plot(self.dt[:t], self.alpha[:t], 'r')
            plt.xlabel('t')
            plt.ylabel('Population')
            plt.legend(["gamma", "beta", "alpha"])
            plt.title("Population change over time")

            # plot a 2d grid of the population
            plt.subplot(2, 1, 1)
            plt.imshow(self.save_AC[t], cmap=cmap, interpolation='nearest')
            plt.colorbar()
            plt.title("t = " + str(t))

            plt.pause(0.1)


    def plot_vs_2d(self, dont_plot):
        if dont_plot == 1:
            x = self.beta
            y = self.alpha
            xlab = "beta"
            ylab = "alpha"
        elif dont_plot == 2:
            x = self.gamma
            y = self.alpha
            xlab = "gamma"
            ylab = "alpha"
        else:
            x = self.gamma
            y = self.beta
            xlab = "gamma"
            ylab = "beta"

        t = self.dt

        plt.scatter(x, y, c=t, cmap="summer")
        plt.colorbar(label="time")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()


    def plot_population(self):
        plt.plot(self.dt, self.gamma,self.dt,self.beta,self.dt,self.alpha)
        plt.xlabel('t')
        plt.ylabel('Population')

        plt.legend(["gamma", "beta", "alpha"])
        plt.title("Population change over time")
        plt.show()

    def plot_vs_3d(self):
        # Creating figure
        #fig = plt.figure(figsize=(10, 7))
        #ax = plt.axes(projection="3d")
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        [x,y,z] = [self.gamma, self.beta, self.alpha]
        ax.scatter(x, y, z, s=50, c='k')
        # Creating plot
        #ax.scatter3D(self.gamma, self.beta, self.alpha, color="green")

        ax.plot(x, z, 'ro', zdir='y', zs=max(y)+10, alpha=0.2)
        ax.plot(y, z, 'go', zdir='x', zs=min(x)-10, alpha=0.2)
        ax.plot(x, y, 'bo', zdir='z', zs=min(z)-10, alpha=0.2)

        ax.set_xlim([min(x)-10, max(x)+10])
        ax.set_ylim([min(y)-10, max(y)+10])
        ax.set_zlim([min(z)-10, max(z)+10])

        plt.title("")
        ax.set_xlabel('$Gamma$')
        ax.set_ylabel('$Beta$')
        ax.set_zlabel('$Alpha$')
        # show plot
        plt.show()

    def integrate_stable(self):
        """integrator funtion, will perform all steps, and break if any species dies out"""
        bad = 0
        for t in range(self.T + 1):
            if t == self.T_intro:
                if self.intro:
                    self.introduce()

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
            #print(self)

            if self.two:
                if gamma == 0 or alpha == 0:
                    bad = 1
                    break
            else:
                if gamma == 0 or alpha == 0 or beta == 0:
                    bad = 1
                    break

        return t, bad

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



### Below are functions to use. If mod=True system S4 will be used. If mod=False S1 will be used.
# To change this, one must change the PP-class. Feel free to do so.
# When plotting the S0 (gamma/alpha) set mod=True to get similar results to the text


def introduce(L=40, T=200, g=100, b=0, a=100, mod=True):
    """Function that will introduce the n=20 agents, of the one that is initially set to zero after t=20 seconds.
    If you want to change t or n, change the following line:
    PP = PPsystem(init_cond, L, T, mod=mod, animate=True, intro=intro, T_intro=XXX, n_intro=XXX). Ugly solution since this is not used in the code """
    if g == 0:
        intro = 1
    elif b == 0:
        intro = 2
    else:
        intro = 3
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T, mod=mod, animate=True, intro=intro)
    PP.integrate()
    PP.anim_and_population_plot()



def plot_VS(L=40, T=200, g=100, b=100, a=100, mod=True):
    """Function that plots population portrait, set one of the populations to zero for 2d-plot."""
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T, mod=mod, animate=True)
    if g == 0:
        PP.plot_vs_2d(dont_plot=1)
    elif b == 0:
        PP.plot_vs_2d(dont_plot=2)
    elif a == 0:
        PP.plot_vs_2d(dont_plot=3)
    else:
        PP.plot_vs_3d()

def animate(L=40, T=200, g=100, b=100, a=100, mod=True):
    """show animation and population plot"""
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T, mod=mod, animate=True)
    PP.integrate()
    PP.anim_and_population_plot()

def plot_pop(L=40, T=200, g=100, b=100, a=100, mod=True, until_extinction=False):
    """plot only population plot, animate will be set to false, to save memory/time.
    if until_extinction=True the integration will finish if one of the species dies out. <-- This will only work if all species are alive at t=0"""
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T, mod=mod, animate=False)
    if until_extinction:
        PP.integrate_stable()
    else:
        PP.integrate()
    PP.plot_population()


def stability(L=20, T=1000, g=25, b=25, a=25, mod=False, VB=0.9, two = False):
    """returns how long this system survived"""
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T, mod=mod, animate=False, VB=VB, two=two)
    end, bad = PP.integrate_stable()
    return end, bad



n = 50

M_vec = []
av_time = [[],[],[],[]]
makes_it = [[],[],[],[]]
logi = []

for M in range(5,35):
    ini = round(M*M/16)
    END = [[],[],[],[]]
    BAD = [[],[],[],[]]
    for i in range(n):
        #S0
        end_time, bac_agent = stability(M,1000,ini,0,ini,True,0.9,True)
        END[0].append(end_time)
        BAD[0].append(bac_agent)

        #S2
        end_time, bac_agent = stability(M,1000,ini,ini,ini,False,0.9)
        END[1].append(end_time)
        BAD[1].append(bac_agent)

        #S3 0.99
        end_time, bac_agent = stability(M,1000,ini,ini,ini,True,0.99)
        END[2].append(end_time)
        BAD[2].append(bac_agent)

        #S4
        end_time, bac_agent = stability(M,1000,ini,ini,ini,True,0.9)
        END[3].append(end_time)
        BAD[3].append(bac_agent)

    M_vec.append(M)
    logi.append(0.04 * M ** 3)
    for k in range(4):
        av_time[k].append(sum(END[k])/n)
        makes_it[k].append(BAD[k].count(0)/n)



plt.figure()
plt.loglog(M_vec,av_time[0], "x-", M_vec,av_time[1], "x-", M_vec,av_time[2], "x-", M_vec, av_time[3], "x-", M_vec, logi)

plt.xlabel('M')
plt.ylabel('[%]')

plt.legend(["$\mathcal{S}_1$", "$\mathcal{S}_2$", "$\mathcal{S}_3$", "$\mathcal{S}_4$", "$c \cdot M^3$"])
#plt.title("Average lenght of time before first species faces extinction")
plt.show()





print("s0 - times", av_time[0])
print("s0 - makes it to 1000", makes_it[0])

print("s2 - times", av_time[1])
print("s2 - makes it to 1000", makes_it[1])

print("s3 - times", av_time[2])
print("s3 - makes it to 1000", makes_it[2])

print("s4 - times", av_time[3])
print("s4 - makes it to 1000", makes_it[3])

print("M", M_vec)