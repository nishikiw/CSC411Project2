from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import scipy.stats


def main():
    theta = array([-3, 1.5])
    N = 50
    sigma = 100
    gen_lin_data_1d(theta, N, sigma)


def f_project1(x, y, theta):
    """Cost function for part 6."""
    
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (dot(theta.T,x) - y) ** 2)


def df_project1(x, y, theta):
    """Derivative of the cost function."""
    
    x = vstack( (ones((1, x.shape[1])), x))
    return 2 * dot(x, (dot(theta.T, x) - y).T)
  

def f_project2(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return -sum(y*log(p)) 
    
    
def df_project2(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return dot(x, (p - y).T)


def lin_combin(w, x):
    o = dot(w.T, x)
    return softmax(o)
    
    
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def grad_descent(f, df, x, y, init_t, alpha, max_iter):
    """
    Gradient descent function borrowed from
    http://www.cs.toronto.edu/~guerzhoy/411/lec/W02/python/linreg.html
    """
    
    EPS = 1e-8   #EPS = 10**(-8)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        # if iter % 500 == 0:
        #     print "Iter", iter
        #     print "f(x) = %.2f" % (f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t


def plot_line(theta, x_min, x_max, color, label):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)


def test_performance(x, theta, y):
    correct = 0
    x = vstack( (ones((1, x.shape[1])), x))
    result = dot(theta.T, x)
    for i in range(0, y.shape[1]):
        if y[argmax(result[:, i]), i] == 1:
            correct += 1
    return correct * 100 / y.shape[1]


def gen_lin_data_1d(theta, N, sigma):

    #####################################################
    # Actual data
    #random.seed(0)
    x1_raw = 100*(random.random((N))-.5)
    #random.seed(1)
    x2_raw = 100*(random.random((N))-.5)
    
    # x1_test = vstack((    ones_like(x1_raw),
    #                 x1_raw,
    #                 ))
                    
    #x2_raw = dot(theta, x1_test) + scipy.stats.norm.rvs(scale= sigma,size=N)
    x = vstack((x1_raw, x2_raw))
    
    y = zeros((4, len(x1_raw)))
    colors = array([])
    color1 = array([1., 0., 0., 1])
    color2 = array([0., 1., 0., 1])
    color3 = array([0., 0., 1., 1])
    color4 = array([1., 1., 1., 1])
    
    # Get y.
    for i in range(0, len(x1_raw)):
        if x2_raw[i] > 0:
            if x1_raw[i] > 0:
                y[0, i] = 1
                y[1, i] = 0
                y[2, i] = 0
                y[3, i] = 0
                if len(colors) == 0:
                    colors = color1
                else:
                    colors = vstack((colors, color1))
            else:
                y[0, i] = 0
                y[1, i] = 1
                y[2, i] = 0
                y[3, i] = 0
                if len(colors) == 0:
                    colors = color2
                else:
                    colors = vstack((colors, color2))
        else:
            if x1_raw[i] > 0:
                y[0, i] = 0
                y[1, i] = 0
                y[2, i] = 1
                y[3, i] = 0
                if len(colors) == 0:
                    colors = color3
                else:
                    colors = vstack((colors, color3))
            else:
                y[0, i] = 0
                y[1, i] = 0
                y[2, i] = 0
                y[3, i] = 1
                if len(colors) == 0:
                    colors = color4
                else:
                    colors = vstack((colors, color4))
    #y = dot(theta, x) + scipy.stats.norm.rvs(scale= sigma,size=N)
    
    
    
    scatter(x1_raw, x2_raw, c=colors)
    #####################################################
    # Actual generating process
    #
    #plot_line(theta, -70, 70, "b", "boundary")
    
    # Plot axis.
    axhline(y=0, color='k', label = "x")
    axvline(x=0, color='k', label="y")
    
    #######################################################
    # Least squares solution
    #
    
    #theta_hat = dot(linalg.inv(dot(x, x.T)), dot(x, y.T))
    #plot_line(theta_hat, -70, 70, "g", "Maximum Likelihood Solution")
    
    #random.seed(1)
    init_t = random.normal(0.0, 1.0, (x.shape[0]+1, y.shape[0]))/math.sqrt((x.shape[0]+1) * y.shape[0])
    max_iter = 30000
    alpha1 = 1e-5
    theta_p1 = grad_descent(f_project1, df_project1, x, y, init_t, alpha1, max_iter)
    performance_p1 = test_performance(x, theta_p1, y)
    print("Project 1 performance is "+str(performance_p1)+"%")
    
    alpha2 = 1e-5
    theta_p2 = grad_descent(f_project2, df_project2, x, y, init_t, alpha2, max_iter)
    performance_p2 = test_performance(x, theta_p2, y)
    print("Project 2 performance is "+str(performance_p2)+"%")
    
        

    legend(loc = 1)
    xlim([-70, 70])
    ylim([-100, 100])
    savefig("part5.png")
    

if __name__ == '__main__':
    main()