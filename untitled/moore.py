import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from matplotlib import animation

# North = down, South = up, East = right, West = left - thats how i happened to write it.


class PPsystem:

    def __init__(self, init_cond, L, T, r_g=2, r_a=2, r_b = 10, mod = False, intro = False, T_intro = 20, n_intro = 20):
        # initialize the AC-system

        self.mod = mod #if true - use the critter-modification
        self.intro = intro
        self.T_intro = T_intro
        self.n_intro = n_intro

        # param
        self.b_g = 0.6 #(1-p) i.e. b_g = 0.4
        self.d_ga = 0.6
        self.d_gb = 0.4#0.2

        self.b_b = 0.5#0.4
        self.d_b = 0.3 #0 #0.3
        self.d_ba = 0.4

        self.d_vb = 0.999 #(1-p) i.e. d_vb=0.1

        self.b_a = 0.3
        self.d_a = 0.2



        self.proc_b = 0.9
        self.proc_a = 0.9


        # constants
        self.L = L #size
        self.r_g = r_g #depth of moore neighborhood
        self.r_a = r_a
        self.r_b = r_b
        self.T = T


        #will change

        self.dt = [0]

        self.AC = init_cond  # initial matrix
        self.gamma = []
        self.beta = []
        self.alpha = []
        self.save_AC = []

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
        s = self.intro
        a = self.n_intro
        i = 0
        j = 0
        while i < a:
            j += 1
            #x = random.randint(0, self.n_intro)
            #y = random.randint(0, self.n_intro)
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            if self.AC[x][y] == 0:
                self.AC[x][y] = s
                i += 1
            if j > self.L*self.L:
                i = a


    def attack(self,x,y):
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




    def get_quarters(self, x,y,cell): #returns moore-neighbors for all 4 quarters
        n_NSEW = [[], [], [], []]

        if cell == 1:
            R = self.r_g
        elif cell == 2:
            R = self.r_b
        else:
            R = self.r_a

        for i in range(1, R + 1):
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
                    n_NSEW, NSEW = self.get_quarters(x, y,cell)
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


    def integrate_print(self):
        print("initial conditions:")
        for lst in self.AC: #print initial conditions
            print(lst)
        for t in range(self.T + 1):
            self.dt.append(t+1)

            self.reaction()
            print("reaction phase, t = ", self.dt[t])
            for lst in self.AC:
                print(lst)
            self.movement()
            print("movement phase, t = ", self.dt[t])
            gamma = 0
            beta = 0
            alpha = 0
            for lst in self.AC:
                gamma += lst.count(1)
                beta += lst.count(2)
                alpha += lst.count(3)
                print(lst)
            self.gamma.append(gamma)
            self.beta.append(beta)
            self.alpha.append(alpha)

    def integrate(self):
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

            self.save_AC.append(self.AC)

    def animate(self):
        """Function to animate the AC, this function will"""
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

            # plot a 2d grid of the population
            plt.clf()


            cmap = (mpl.colors.ListedColormap(['white', 'green', 'blue', 'cyan'])
                    .with_extremes(over='0.25', under='0.75'))

            bounds = [0, 1, 2, 3,4]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

            #plt.imshow(self.AC, cmap=cmap, interpolation='nearest')
            #plt.title("t = " + str(t))
            #plt.pause(0.1)
            #show animation of self.AC

            plt.subplot(2, 1, 2)
            plt.plot(self.dt, self.gamma, 'g')
            plt.plot(self.dt, self.beta, 'b')
            plt.plot(self.dt, self.alpha, 'r')
            plt.xlabel('t')
            plt.ylabel('Population')
            plt.legend(["gamma", "beta", "alpha"])
            plt.title("Population change over time")

            # plot a 2d grid of the population
            plt.subplot(2, 1, 1)
            plt.imshow(self.AC, cmap=cmap, interpolation='nearest')
            plt.colorbar()
            plt.title("t = " + str(t))

            plt.pause(0.1)


    def animate_test(self):
        for t in range(self.T + 1):
            plt.clf()

            cmap = (mpl.colors.ListedColormap(['white', 'green', 'blue', 'red'])
                    .with_extremes(over='0.25', under='0.75'))

            bounds = [0, 1, 2, 3,4]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

            #plt.imshow(self.AC, cmap=cmap, interpolation='nearest')
            #plt.title("t = " + str(t))
            #plt.pause(0.1)
            #show animation of self.AC

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

    def plot_population(self):

        # plt.plot(self.dt, self.gamma,self.dt,self.beta,self.dt,self.alpha)

        leg = []

        if sum(self.gamma) != 0:
            leg.append("Gamma")
            plt.plot(self.dt, self.gamma, "g")
        if sum(self.beta) != 0:
            leg.append("Beta")
            plt.plot(self.dt, self.beta, "b")
        if sum(self.alpha) != 0:
            leg.append("Alpha")
            plt.plot(self.dt, self.alpha, "r")

        plt.xlabel('t')
        plt.ylabel('Population')
        # plt.legend(["gamma", "beta", "alpha"])
        plt.legend(leg)
        plt.title("Population change over time")
        plt.show()


    def plot_vs_3d(self):

        # Creating figure
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(self.gamma, self.beta, self.alpha, color="green")
        plt.title("")
        ax.set_xlabel('$\gamma$')
        ax.set_ylabel('$ beta$')
        ax.set_zlabel('$alpha$')
        # show plot
        plt.show()








def create_init(L = 10, g=3, b=3, a=3):
    init_cond = [0] * L
    for i in range(len(init_cond)):
        init_cond[i] = [0] * L

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






def testing():
    init_cond = [[1, 0, 1, 2, 1], [1, 1, 0, 1, 2], [1, 1, 0, 0, 3], [2, 1, 0, 0, 0], [2, 1, 0, 0, 0]]
    # init_cond = [[1,0,1],[0,1,0],[2,2,1]]
    L = 5
    T = 10
    PP = PPsystem(init_cond, L, T)

    PP.integrate_print()
    #PP.plot_population()

#testing()


def just_gamma():
    [L,T,g,b,a] = [40,300,50,0,0]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    PP.plot_population()

def just_alpha():
    [L,T,g,b,a] = [40,300,0,0,50]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    PP.plot_population()

def just_beta():
    [L,T,g,b,a] = [40,300,0,100,0]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    #PP.integrate()
    #PP.plot_population()
    PP.animate()
def no_beta():
    [L,T,g,b,a] = [40,100,1000,0,600]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    #PP.plot_population()
    PP.animate_test()

def no_alpha():
    [L,T,g,b,a] = [40,100,70,50,0]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    PP.plot_population()

def no_gamma():
    [L,T,g,b,a] = [40,300,0,70,50]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    #PP.plot_population()
    #PP.plot_vs_3d()
    PP.animate_test()


def all_three():
    [L,T,g,b,a] = [40,200,50,20,10]
    init_cond = create_init(L,g,b,a)
    PP = PPsystem(init_cond, L, T)
    PP.integrate()
    PP.plot_population()
    #PP.plot_vs_3d()
    #PP.animate_test()


def moore():
    [L, T, g, b, a] = [40, 300, 100, 100, 100]
    plot_list = []
    rad = []
    for radius in range(2,10):
        deaths = []
        for i in range(10):
            init_cond = create_init(L, g, b, a)
            PP = PPsystem(init_cond, L, T, r_b=radius)
            PP.integrate()
            death = 0
            for pop in PP.beta:
                if pop != 0:
                    death += 1
            deaths.append(death)
        print(sum(deaths)/10)
        plot_list.append(sum(deaths)/10)
        rad.append(radius)
    plt.plot(rad, plot_list)
    plt.xlabel('Moore radius of Beta')
    plt.ylabel('Time of death')
    # plt.legend(["gamma", "beta", "alpha"])
    plt.title("Giving Beta an edge using Moore radius")
    plt.show()




#just_gamma()
#just_alpha()
#just_beta()
#no_beta()
#no_alpha()
#no_gamma()
#all_three()
moore()