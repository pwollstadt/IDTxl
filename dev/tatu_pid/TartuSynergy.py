# This is file TartuSynergy.py  ( Python 3.x )
# Uses numpy, cvxopt, and gurobipy
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Main function: solve_PDF()
# ==========================
#
# Takes a joint probability density function of three RVs X,Y,Z, and
# computes synergetic and shared information parts of I( X ; YZ ).
#
# How does it do that?
# --------------------
#
# Iteratively solves Convex Programs (CVXOPT), then attempts to ``tidy
# up'' the solution by setting variables which are ``dirty zeros'' to
# actual zeros. For deciding which variables are dirty zeros, it
# solves a sequence of Linear Programs (Gurobi).
#
# Usage
# -----
#
# def solve_PDF(pdf, true_pdf=None, true_result=None, true_CI=None, true_SI=None,
#               feas_eps  =1.e-10, feas_eps_2 =1.e-6,
#               kkt_eps   =1.e-5,  kkt_eps_2  =.01,
#               kkt_search_eps =.5, max_zero_probability=1.e-5,
#               verbose=False):
#
# PARAMETERS
#
# pdf                  The probability density function.
#                      Dictionary mapping triples to real numbers.
#                      The first in the triple is the `dependent' RV,
#                      i.e. P( X=x, Y=y, Z=z )  = pdf[ x,y,z ] .
# true_pdf             If `pdf' is an approximation of a known `true' PDF,
#                      you can give the true pdf here, to get more detailed
#                      info (if verbose is True)
# true_result          If the `right' optimal PDF is known, give it here to
#                      have more detailed output (if verbose is True)
# true_CI              If the synergetic info is known, give it here to have
#                      more detailed output (if verbose is True)
# true_SI              If the shared information is known, give it here to
#                      have more detailed output (if verbose is True)
# feas_eps             Maximal sum of violations of marginal equations that is
#                      desirable.  Default: 1.e-10
# kkt_eps              Maximal maximum of violations of KKT-system constraints
#                      that is desirable. Default: 1.e-5
# feas_eps_2           If no further improvement can be found,
#                      this violation of the marginal equations is acceptable.
#                      Default: 1.e-6
# kkt_eps_2            If no further improvement can be found,
#                      this KKT-system maximum constraint violation is
#                      acceptable. Default: .01
# kkt_search_eps       During iterative search for improvements,
#                      when variables are scanned to be set to 0,
#                      if the KKT-system maximum violation drops by this
#                      factor, that improvement is taken. Default: .5
# max_zero_probability During iterative search for improvements,
#                      when variables are scanned to be set to 0,
#                      this is the largest probability which can be considered
#                      to be really 0.  Default: 1.e-5
# verbose              If True, produce lots of intermediate output.
#                      Default: False
#
# RETURN VALUE
#
# The function returns a quadruple:  opt_pdf,feas,kkt,CI,SI
#
# opt_pdf              The optimal (or best guess) PDF found
# feas                 Sum of violations of the marginal equations of opt_pdf
#                      Max of violations of the KKT-system constraints
# CI, SI               Synergetic and shared information
#
#^ END OF DOCUMENTATION
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from cvxopt import solvers, matrix, spmatrix, spdiag, log
import numpy
from math import sqrt

# ******************************************************************************************************************************************************
# *  M a r g i n a l s
def marginal_xy(p):
    marg = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if (x,y) in marg.keys():    marg[(x,y)] += r
        else:                       marg[(x,y)] =  r
    return marg

def marginal_xz(p):
    marg = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if (x,z) in marg.keys():   marg[(x,z)] += r
        else:                      marg[(x,z)] =  r
    return marg


def marginal_yz(p):
    marg = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if (y,z) in marg.keys():    marg[(y,z)] += r
        else:                       marg[(y,z)] =  r
    return marg
def marginal_yz_with_cutoff(p, ZERO=1.e-1000):
    marg = dict()
    for xyz,r in p.items():
        if r>ZERO:
            x,y,z = xyz
            if (y,z) in marg.keys():    marg[(y,z)] += r
            else:                       marg[(y,z)] =  r
    return marg


def marginal_x(p):
    marg  = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if x in marg.keys():   marg[x] += r
        else:                  marg[x] =  r
    return marg

def marginal_y(p):
    marg  = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if y in marg.keys():   marg[y] += r
        else:                  marg[y] =  r
    return marg

def marginal_z(p):
    marg  = dict()
    for xyz,r in p.items():
        x,y,z = xyz
        if z in marg.keys():   marg[z] += r
        else:                  marg[z] =  r
    return marg

#^ marginals
# ******************************************************************************************************************************************************


# ******************************************************************************************************************************************************
# *  C L A S S   for the computation of unique information
class Compute_UI:
    def __init__(self, marg_xy, marg_xz, _set_to_zero=set()):
        # marg_xy is a dictionary     (x,y) --> positive double
        # marg_xz is a dictionary     (x,z) --> positive double

        self.orig_marg_xy = None
        self.orig_marg_xz = None
        self.var_eps      = None
        self.q_xy         = None
        self.q_xz         = None
        self.var_idx      = None
        self.X            = None
        self.Y            = None
        self.Z            = None
        self.A            = None
        self.b            = None
        self.marg_of_idx  = None  # triples xyz -- the missing one is None
        self.create_equations_called = False
        self.G            = None
        self.h            = None
        self.create_ieqs_called = False
        self.p_0          = None # initial solution
        self.solver_ret   = None
        self.p_final      = None
        self.set_to_zero  = _set_to_zero

        # Options:
        self.expensive_initial_solution = False
        self.verbose_output             = False

        # Actual code:
        self.orig_marg_xy = dict(marg_xy)
        self.orig_marg_xz = dict(marg_xz)
        self.X = set( [ x   for x,y in self.orig_marg_xy.keys() ] + [ x   for x,z in self.orig_marg_xz.keys() ] )
        self.Y = set( [  y  for x,y in self.orig_marg_xy.keys() ] )
        self.Z = set(                                               [  z  for x,z in self.orig_marg_xz.keys() ] )
    # __init__()

    # tidy_up_distrib():
    def tidy_up_distrib(self,p, eps):
        # returns a tidied-up copy of p: every entry smaller than eps is treated as 0.
        p_new = dict()
        one = 0.
        for x,r in p.items():
            if r>=eps:
                p_new[x] = r
                one += r
        # Re-normalize --- Should I drop this ??? ?!?
        for x,r in p_new.items():
            p_new[x] = r / one;

        return p_new
    #^ tidy_up_distrib()

    # create_equations():
    def create_equations(self):
        # The point is that all the double values are considered non-zero
        # This function
        #  - creates the sets X,Y,Z
        #  - creates the dictionary var_idx: variables --> index of the variable
        #  - creates the matrices A,b: two types of marginal equations: p(x,y,*)==p_xy(x,y), p(x,*,z)==p_xz(x,z)
        #  - A has full rank: unneeded eqns are thrown out
        if self.create_equations_called:
            print("Some dork called create_equations twice...")
            exit(1)
        self.create_equations_called = True

        count_vars = 0
        self.var_idx    = dict()
        for x in self.X:
            for y in self.Y:
                if (x,y) in self.q_xy.keys():
                    for z in self.Z:
                        if (x,y,z) not in self.set_to_zero  and  (x,z) in self.q_xz.keys():
                            self.var_idx[ (x,y,z) ] = count_vars
                            count_vars += 1

        list_b           = [] # list of RHSs
        list_At          = [] # coefficient matrix in row-major (need column-major later)
        list_At_throwout = [] # DEBUG: omited equations go here---then we check whether the ranks are equal
        numo_thrownout   = 0

        self.marg_of_idx = []
        # xy-marginal equations
        for xy,rhs in self.q_xy.items():
            x,y = xy
            a = [ 0   for xyz in self.var_idx.keys() ] # initialize the whole row with 0
            for z in self.Z:
                if (x,y,z) in self.var_idx.keys():
                    i  = self.var_idx[ (x,y,z) ]
                    a[i] = 1.
            # list_At += a # splice !
            # list_b.append( rhs )
            # self.marg_of_idx.append(  (x,y,None)  )

            # We test if adding this equation increases the rank.
            # Because of the deleted variables (in set_to_zero), I don't know of a better way to do this...
            tmp_At = matrix( list_At+a, ( len(self.var_idx),  len(list_b)+1 ), 'd' )
            if numpy.linalg.matrix_rank( tmp_At ) > len(list_b):
                list_At += a # splice !
                list_b.append( rhs )
                self.marg_of_idx.append(  (x,None,z)  )

        # xz-marginal equations
        for xz,rhs in self.q_xz.items():
            x,z = xz
            a = [ 0   for xyz in self.var_idx.keys() ] # initialize the whole row with 0
            for y in self.Y:
                if (x,y,z) in self.var_idx.keys():
                    i = self.var_idx[ (x,y,z) ]
                    a[i] = 1.
            # Rank-check again:
            tmp_At = matrix( list_At+a, ( len(self.var_idx),  len(list_b)+1 ), 'd' )
            if numpy.linalg.matrix_rank( tmp_At ) > len(list_b):
                list_At += a # splice !
                list_b.append( rhs )
                self.marg_of_idx.append(  (x,None,z)  )

        # Now we create the cvxop matrix.
        # Keep in mind: in list_At, there's one equation too many, they are linearly dependent.
        self.b  = matrix( list_b,                   ( len(list_b),        1                          ), 'd' )
        At      = matrix( list_At,                  ( len(self.var_idx),  len(list_b)                ), 'd' )
        self.A = At.T
        rk = numpy.linalg.matrix_rank(self.A)
        if ( rk != len(list_b) ):
            print("BUG: There's something wrong with the rank of the coefficient matrix: it is ",rk," it should be ",len(list_b))
            exit(1)
        dim_space = len(self.var_idx)-rk
        print("Solution space has dimension ",dim_space)
    #^ create_equations()

    # create_ieqs():
    def create_ieqs(self):
        if not self.create_equations_called:
            print("You have to call create_equations() before calling create_ieqs()")
            exit(1)
        if self.create_ieqs_called:
            print("Some dork called create_ieqs() twice...")
            exit(1)
        self.create_ieqs_called = True
        self.G = spdiag( matrix( -1., (len(self.var_idx),1), 'd' ) )
        self.h = matrix( 0, (len(self.var_idx),1), 'd' )
    #^ create_ieqs()

    # CALLBACK fn for computing  f, grad f, Hess f
    def callback(self,p=None, zz=None):
        N = len(self.var_idx)
        if p is None:
            list_p_0 = [ 0.   for xyz in self.var_idx.keys() ]
            for xyz,i in self.var_idx.items():
                list_p_0[i] = 1. # self.p_0[xyz]
            # This is returns the starting solution for the iterative solution of the CP --- this is the 1st point to experiment with other distributions with the same marginals.
            return 0, matrix(list_p_0, (N,1), 'd' )

        # check if p is in the feasible region for the objective function:
        if min(p) <= 0 or max(p) > 1:
            return None

        p_dict = dict( (xyz,p[i]) for xyz,i in self.var_idx.items() )
        p_yz = marginal_yz(p_dict)

        # Compute f(p)
        f = 0
        for xyz,i in self.var_idx.items():
            x,y,z = xyz
            if p[i] > 0: f += p[i]*log(p[i]/p_yz[y,z])
#            if p[i] > 1.e-10: f += p[i]*log(p[i]/p_yz[y,z])

        # Compute gradient-transpose Df(p)
        list_Df = [ 0. for xyz in self.var_idx.keys() ]
        for xyz,i in self.var_idx.items():
            x,y,z = xyz
            if p[i] > 0:  list_Df[i] = log( p[i] / p_yz[y,z] )
#            if p[i] > 1.e-30:  list_Df[i] = log( p[i] / p_yz[y,z] )
        Df = matrix(list_Df, (1,N), 'd')

        if zz is None:
            return f,Df

        # Compute zz[0] * Hess f
        # This will be a sparse matrix
        entries = []
        rows    = []
        columns = []
        for xyz,i in self.var_idx.items():
            x,y,z = xyz
            p_yz__x = p_yz[y,z] - p[i] # sum_{* \ne x} p(*,y,z).
            for x_ in self.X:
                if x_==x: # diagonal
                    rows.append( i )
                    columns.append( i )
                    tmp_quot = zz[0] * p_yz__x / p_yz[y,z] # 1/p[x,y,z] - 1/p[*,y,z] = ( p[*,y,z] - p[x,y,z] )/( p[*,y,z] p[x,y,z] )
                    if p[i] > 0 * tmp_quot:   entries.append( tmp_quot / p[i] )
#                    if p[i] > 1.e-100 * tmp_quot:   entries.append( tmp_quot / p[i] )
                    else:
                        print("TROUBLE computing Hessian (diagonal)")
                        entries.append( 0. )
                else: # off diagonal
                    if (x_,y,z) in self.var_idx:
                        j = self.var_idx[ (x_,y,z) ]
                        val = - zz[0] / p_yz[y,z]
                        rows.append( i )
                        columns.append( j )
                        entries.append( val )
                # if diagonal
            # for x_
        # for xyz,i

        zH = spmatrix( entries, rows, columns, (N,N), 'd')
        if self.verbose_output: print("p=",list(p))
        # # print(numpy.squeeze(zH/zz[0]))
        # print("var_idx=",self.var_idx)
        # print("p=",p_dict)
        # print("p_yz=",p_yz)
        # print("H=",zH)
        # # print("Eigen values of Hess f (p) are ",numpy.linalg.eigvals(matrix(zH)),"[",N,"]")
        # print("Eigen vectors:\n", numpy.linalg.eigh(matrix(zH)))
        return f,Df,zH
    #^ callback()


    # make_initial_solution()
    def make_initial_solution(self, orig_q_0=None):
        # make initial solution q_0
        self.p_0 = dict()
        if orig_q_0==None: # no initial solution provided. I'm gonna get some "minimum entropy" thing
            if self.expensive_initial_solution:
                # initial solution is the one minimizing || p ||^2
                N = len(self.var_idx)

                Q = 2*spdiag( matrix( 1., (N,1), 'd' ) )
                q = matrix( 0., (N,1), 'd' )

                qp_solver = solvers.qp(Q, q, self.G, self.h, self.A, self.b)

                # for xyz,i in self.var_idx.items():
                #     self.p_0[xyz] = max(0, qp_solver['x'][i] )
                print("Min 2-norm: ",list( qp_solver['x'] ))
            else:
                for xyz in self.var_idx.keys():
                    self.p_0[xyz] = 1.
        else:
            for xyz in self.var_idx.keys():
                if xyz in orig_q_0:
                    self.p_0[xyz] = orig_q_0[xyz]
                else:
                    self.p_0[xyz] = 0.
    #^ make_initial_solution()

    # solve_it():
    def solve_it(self, _eps=1.e-10):
        self.var_eps = _eps
        self.q_xy = self.tidy_up_distrib(self.orig_marg_xy, self.var_eps)
        self.q_xz = self.tidy_up_distrib(self.orig_marg_xz, self.var_eps)

        if self.verbose_output:
            print("q_xy=",self.q_xy)
            print("q_xz=",self.q_xz)

        self.create_equations()
        self.create_ieqs()
        self.make_initial_solution()
        if self.verbose_output: print(self.p_0)

        self.solver_ret   = solvers.cp(self.callback, G=self.G, h=self.h, A=self.A, b=self.b)
        print("Solver terminated with status ",self.solver_ret['status'])

        self.p_final = dict()
        for xyz,i in self.var_idx.items():
            self.p_final[xyz] = self.solver_ret['x'][i]
        return self.p_final
    #^ solve_it()

    #****************************************************************************************************

    # check_feasible()
    def check_feasible(self, p, use_cleaned_up_margs=False):
        p_xy = marginal_xy(p)
        p_xz = marginal_xz(p)
        thesum = 0.

        if use_cleaned_up_margs:
            for x in self.X:
                for y in self.Y:
                    xy = (x,y)
                    if xy in p_xy.keys():      p_val = p_xy[xy]
                    else:                      p_val = 0.
                    if xy in self.q_xy.keys():         q_val = self.q_xy[xy]
                    else:                              q_val = 0.
                    thesum += abs( p_val - q_val )
            for x in self.X:
                for z in self.Z:
                    xz = (x,z)
                    if xz in p_xz.keys():      p_val = p_xz[xz]
                    else:                      p_val = 0.
                    if xz in self.q_xz.keys():         q_val = self.q_xz[xz]
                    else:                              q_val = 0.
                    thesum += abs( p_val - q_val )
        else: # if NOT use cleaned up margs:
            for x in self.X:
                for y in self.Y:
                    xy = (x,y)
                    if xy in p_xy.keys():      p_val = p_xy[xy]
                    else:                      p_val = 0.
                    if xy in self.orig_marg_xy.keys(): q_val = self.orig_marg_xy[xy]
                    else:                              q_val = 0.
                    thesum += abs( p_val - q_val )
            for x in self.X:
                for z in self.Z:
                    xz = (x,z)
                    if xz in p_xz.keys():      p_val = p_xz[xz]
                    else:                      p_val = 0.
                    if xz in self.orig_marg_xz.keys(): q_val = self.orig_marg_xz[xz]
                    else:                              q_val = 0.
                    thesum += abs( p_val - q_val )

        return thesum
    #^ check_feasible()

    def check_KKT_sol(self,p,why):
        p_yz = marginal_yz(p)
        viol = -1.

        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    try:               idx_xy = self.marg_of_idx.index((x,y,None))
                    except ValueError: idx_xy = None
                    try:               idx_xz = self.marg_of_idx.index((x,None,z))
                    except ValueError: idx_xz = None

                    if idx_xy == None:  lambda_xy = None
                    else:               lambda_xy = why[idx_xy]
                    if idx_xz == None:  mu_xz = None
                    else:               mu_xz = why[idx_xz]

                    if (x,y,z) in p.keys() and p[x,y,z] > 0:
                        # equation
                        p_xzy = p[x,y,z]
                        rhs = -log(p_xzy/p_yz[y,z])  # >= 0
                        viol = max(viol, abs( lambda_xy + mu_xz - rhs ) )
                    else:
                        # inequality
                        if (y,z) in p_yz.keys() and p_yz[y,z] > 0:
                            if lambda_xy==None or mu_xz==None:
                                pass
                            else:
                                rhs = 1.e400
                                viol = max(viol, rhs - (lambda_xy + mu_xz) )
                        else:
                            rhs = 0.
                            viol = max(viol, rhs - (lambda_xy + mu_xz) )
                    # if
                #^ for z
            #^ for y
        #^ for x
        return viol
    #^ check_KKT_sol()

    class KKT_System:
        import gurobipy as gurobi
        def __init__(self,cui):
            self.cui = cui
            self.model = Compute_UI.KKT_System.gurobi.Model("kkt")
            self.model.params.logToConsole = 0

            # Add the variables:
            self.t_var       = None
            self.lambda_vars = dict()
            self.mu_vars     = dict()

            self.t_var = self.model.addVar(obj=1., name="t")
            for x in self.cui.X:
                for y in self.cui.Y:
                    self.lambda_vars[x,y] = self.model.addVar(name='lambda(%s,%s)' % (x,y))
            for x in self.cui.X:
                for z in self.cui.Y:
                    self.mu_vars[x,z] = self.model.addVar(name='mu(%s,%s)' % (x,z))
            self.model.update()

            # Add the constraints:
            self.constr_leq = dict()
            self.constr_geq = dict()
            for x in self.cui.X:
                for y in self.cui.Y:
                    for z in self.cui.Z:
                        rhs   = 0
                        self.constr_leq[x,y,z] = self.model.addConstr( self.lambda_vars[x,y] + self.mu_vars[x,z] - rhs <= self.t_var , name='eq-le(%s,%s,%s)' % (x,y,z))
                        self.constr_geq[x,y,z] = self.model.addConstr( self.lambda_vars[x,y] + self.mu_vars[x,z] - rhs >= -self.t_var, name='eq-ge(%s,%s,%s)' % (x,y,z))
                    #^ for z
                #^ for y
            #^ for x
            self.model.update()
            self.first_time=True
        #^ __init__()

        def solve(self, p, ZERO=1.e-1000):
            p_yz = marginal_yz_with_cutoff(p,ZERO)
            if self.first_time:
                self.first_time = False
                for x in self.cui.X:
                    for y in self.cui.Y:
                        for z in self.cui.Z:
                            xyz=x,y,z
                            if xyz in p.keys() and p[xyz] > ZERO:
                                # equation
                                p_xyz = p[xyz]
                                rhs = -log(p_xyz/p_yz[y,z])  # >= 0
                                self.constr_geq[ xyz ].setAttr("rhs", rhs)
                                self.constr_leq[ xyz ].setAttr("rhs", rhs)
                            else:
                                # inequality
                                if xyz in self.constr_leq.keys():
                                    self.model.remove(self.constr_leq[xyz])
                                    del self.constr_leq[ xyz ]
                                if (y,z) in p_yz.keys() and p_yz[y,z] > 0:    rhs = 1.e400       # don't need ZERO here,  marginal_yz_w_cutoff()
                                else:                                         rhs = 0.
                                self.constr_geq[ xyz ].setAttr("rhs", rhs)
                            # if
                        #^ for z
                    #^ for y
                #^ for x
            else: # same, but different loop
                for xyz in p.keys():
                    x,y,z=xyz
                    if p[xyz] > ZERO:
                        # equation
                        p_xyz = p[xyz]
                        rhs = -log(p_xyz/p_yz[y,z])  # >= 0
                        self.constr_leq[ xyz ].setAttr("rhs", rhs)
                        self.constr_geq[ xyz ].setAttr("rhs", rhs)
                    else:
                        # inequality
                        if xyz in self.constr_leq.keys():
                            self.model.remove(self.constr_leq[xyz])
                            del self.constr_leq[ xyz ]
                        if (y,z) in p_yz.keys() and p_yz[y,z] > 0:    rhs = 1.e400       # don't need ZERO here,  marginal_yz_w_cutoff()
                        else:                                         rhs = 0.
                        self.constr_geq[ xyz ].setAttr("rhs", rhs)
                    # if
                #^ for xyz
            #^ if/else  first time
            self.model.update()
            self.model.optimize();
            if self.model.status == Compute_UI.KKT_System.gurobi.GRB.Status.OPTIMAL:
                t = self.t_var.getAttr("x")
                return t
            else:
                return None
        #^ solve()
    #^ class KKT_System

    def solve_KKT_system(self, p, ZERO=1.e-1000):
        import gurobipy as gurobi

        model = gurobi.Model("kkt")
        model.params.logToConsole = 0


        # Add the variables:
        t_var       = None
        lambda_vars = dict()
        mu_vars     = dict()

        t_var = model.addVar(obj=1., name="t")
        for x in self.X:
            for y in self.Y:
                lambda_vars[x,y] = model.addVar(name='lambda(%s,%s)' % (x,y))
        for x in self.X:
            for z in self.Y:
                mu_vars[x,z] = model.addVar(name='mu(%s,%s)' % (x,z))
        model.update()

        # Add the constraints:
        p_yz = marginal_yz_with_cutoff(p,ZERO)

        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    if (x,y,z) in p.keys() and p[x,y,z] > ZERO:
                        # equation
                        p_xyz = p[x,y,z]
                        rhs = -log(p_xyz/p_yz[y,z])  # >= 0
                        model.addConstr( lambda_vars[x,y] + mu_vars[x,z] - rhs <= t_var , name='eqn+(%s,%s,%s)' % (x,y,z))
                        model.addConstr( lambda_vars[x,y] + mu_vars[x,z] - rhs >= -t_var , name='eqn-(%s,%s,%s)' % (x,y,z))
                    else:
                        # inequality
                        if (y,z) in p_yz.keys() and p_yz[y,z] > 0:    rhs = 1.e400       # don't need ZERO here,  marginal_yz_w_cutoff()
                        else:                                         rhs = 0.
                        model.addConstr( lambda_vars[x,y] + mu_vars[x,z] - rhs >= -t_var , name='ieq(%s,%s,%s)' % (x,y,z))
                    # if
                #^ for z
            #^ for y
        #^ for x

        # Run Gurobi:
        model.optimize();

        if model.status == gurobi.GRB.Status.OPTIMAL:
            t = t_var.getAttr("x")
            return t
        else:
            return None
    #^ solve_KKT_system()

    def make_KKT_lp(self,p,filename):
        x_idx = dict()
        for x in self.X:
            x_idx[x] = len(x_idx)
        y_idx = dict()
        for y in self.Y:
            y_idx[y] = len(y_idx)
        z_idx = dict()
        for z in self.Z:
            z_idx[z] = len(z_idx)

        p_yz = marginal_yz(p)

        filecontent = ""
        filecontent += "Minimize\n"
        filecontent += "Obj: t\n"
        filecontent += "Subject To\n"
        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    lhs = ""
                    lhs += " lambda"+str(x_idx[x])+"_"+str(y_idx[y])
                    lhs += " + "
                    lhs += " mu"+str(x_idx[x])+"_"+str(z_idx[z])
                    if (x,y,z) in p.keys() and p[x,y,z] > 0:
                        # equation
                        p_xzy = p[x,y,z]
                        rhs = -1000*log(p_xzy/p_yz[y,z])  # >= 0
                        filecontent += "p("+str(x_idx[x])+","+str(y_idx[y])+","+str(z_idx[z])+"):  "+lhs+"   -t <= "+str(rhs)+"\n"
                        filecontent += "p("+str(x_idx[x])+","+str(y_idx[y])+","+str(z_idx[z])+"):  "+lhs+"   +t >= "+str(rhs)+"\n"
                    else:
                        # inequality
                        if (y,z) in p_yz.keys() and p_yz[y,z] > 0:    rhs = 1.e100
                        else:                                         rhs = 0
                        filecontent += "p("+str(x_idx[x])+","+str(y_idx[y])+","+str(z_idx[z])+"):  "+lhs+"   +t >= "+str(rhs)+"\n"
                #^ for
            #^ for
        #^ for
        filecontent += "END\n"
        print("Writing KKT system to file ",filename)
        with open(filename, 'w') as thefile:
            thefile.write(filecontent)
    #^ make_KKT_lp()

    def try_to_improve_by_LP(self, p):
        import gurobipy as gurobi

        model = gurobi.Model("improve")
        model.params.logToConsole = 0

        p_yz = marginal_yz(p)

        # Add the variables:
        p_xyz_vars = dict()

        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    if (x,y,z) in p.keys() and p[x,y,z] > 0:
                        # equation
                        p_xyz = p[x,y,z]
                        obj = -log(p_xyz/p_yz[y,z])  # >= 0
                    else:
                        # inequality
                        if (y,z) in p_yz.keys() and p_yz[y,z] > 0:    obj = 1.e100
                        else:                                         obj = 0.
                    p_xyz_vars[x,y,z] = model.addVar( obj=rhs, lb=0., name='p(%s,%s,%s)' % (x,y,z))
                    # if
                #^ for z
            #^ for y
        #^ for x

        model.update()

        # Add constraints:
        for x in self.X:
            for y in self.Y:
                lambda_vars[x,y] = model.addVar(name='lambda(%s,%s)' % (x,y))
        for x in self.X:
            for z in self.Y:
                mu_vars[x,z] = model.addVar(name='mu(%s,%s)' % (x,z))
        model.update()

        # Add the constraints:
        p_yz = marginal_yz(p)


        # Run Gurobi:
        model.optimize();

        if model.status == gurobi.GRB.Status.OPTIMAL:
            t = t_var.getAttr("x")
            return t
        else:
            print("There has been a terrible mistake! [try_to_improve_LP()]")
            return 1.e400
    #^ try_to_improve_by_LP()


    def check_guess(self, p, y=None, kkt_filename=""):
        infeas_2nm = self.check_feasible(p)
        if y==None: kkt_viol  = self.solve_KKT_system(p)
        else:       kkt_viol  = self.check_KKT_sol(p,y)
        if kkt_filename!="":
            if kkt_filename==None: kkt_filename = "kkt.lp"
            cui.make_KKT_lp(cui.p_final,kkt_filename)
        return infeas_2nm,kkt_viol
    # check_guess()

    def search_for_kkt_solution(self, p, stop_search_kkt_eps=-1., stop_search_prob_eps=1., show_progress=True):
        min_viol = 1.e1000
        arg_min_viol = None
        max_viol = -1.e1000
        delta    = 1.e1000
        kkt_viol_list = []

        the_ZERO_range = [ t    for xyz,t in p.items() ]
        the_ZERO_range.append(0)
        the_ZERO_range.sort()

        prev_viol = -1.e1000
        if show_progress: print("Searching...",end=" ")
        last_best_guess = None
        last_best_guess_kkt = len(p)
        kktsys = Compute_UI.KKT_System(self)
        for the_ZERO in the_ZERO_range:
            if show_progress: print(the_ZERO,end=":")
            kkt_viol = kktsys.solve(p,the_ZERO)
            if show_progress: print(kkt_viol,end="; ")
            if kkt_viol!=None:  max_viol = max(max_viol, kkt_viol)
            else:               max_viol = 1.e1000
            if kkt_viol!=None:
                if the_ZERO <= stop_search_prob_eps:
                    if the_ZERO > 0  and  kkt_viol < last_best_guess_kkt:
                        last_best_guess     = the_ZERO
                        last_best_guess_kkt = kkt_viol
                    if kkt_viol < min_viol:
                        min_viol     = kkt_viol
                        arg_min_viol = the_ZERO
                        if kkt_viol <= stop_search_kkt_eps:
                            if show_progress: print("[Breaking from search.]",end=" ")
                            break
                    delta     = min(delta, prev_viol-kkt_viol)
                    prev_viol = kkt_viol
            #^ if kkt_viol finite
            kkt_viol_list.append( (the_ZERO,kkt_viol) )
            if show_progress==False and the_ZERO > stop_search_prob_eps:
                break
        #^ for the_ZERO
        if show_progress: print("Done\nMin violation",min_viol,"with the_ZERO=",arg_min_viol,"; non-monotonicity:",-delta," max viol:",max_viol)
        if min_viol > stop_search_kkt_eps:
            if show_progress: print("Oh bummer, that's too much KKT violation. Trying fallback: the_ZERO=",last_best_guess)
            if last_best_guess!=None:  arg_min_viol=last_best_guess
            else:
                return None
        return min_viol,arg_min_viol,-delta,max_viol,kkt_viol_list
    # search_for_kkt_solution()

    #****************************************************************************************************

    def options():
        pass
        # solvers.options['show_progress'] = True, False # turns the output to the screen on or off (default: True).
        # solvers.options['maxiters'     ] = 10          # maximum number of iterations (default: 100).
        # solvers.options['abstol'       ] = 1.e-7       # absolute accuracy (default: 1e-7).
        # solvers.options['reltol'       ] = 1.e-6       # relative accuracy (default: 1e-6).
        # solvers.options['feastol'      ] = 1.e-7       # tolerance for feasibility conditions (default: 1e-7).
        # solvers.options['refinement'   ] = 1           # number of iterative refinement steps when solving KKT equations (default: 1).
    #^ options()

#^ class Compute_UI

def I_X_YZ(p):
    # Mutual information I( X ; YZ )
    p_x = marginal_x(p)
    p_yz = marginal_yz(p)
    mysum = 0
    for xyz,t in p.items():
        x,y,z = xyz
        if t>0:  mysum += t * log( t / ( p_x[x]*p_yz[(y,z)] ) )
    return mysum/log(2)
#^ I_X_YZ()

def I_X_Y(p):
    # Mutual information I( X ; Y )
    p_x  = marginal_x(p)
    p_y  = marginal_y(p)
    p_xy = marginal_xy(p)
    mysum = 0
    for xy,t in p_xy.items():
        x,y = xy
        if t>0:  mysum += t * log( t / ( p_x[x]*p_y[y] ) )
    return mysum/log(2)
#^ I_X_Y()

def cond_I_X_Y__Z(p):
    # Conditional mutual information I( X ; Y | Z )
    p_z  = marginal_z(p)
    p_xz = marginal_xz(p)
    p_yz = marginal_yz(p)
    mysum = 0
    for xyz,t in p.items():
        x,y,z = xyz
        if t>0:  mysum += t * log( ( t * p_z[z] )/( p_xz[(x,z)]*p_yz[(y,z)] ) )
    return mysum/log(2)
#^ cond_I_X_Y__Z()

# Synergistic Information
def wriggle_CI(p,q):
    return I_X_YZ(p) - I_X_YZ(q)
#^ wriggle_CI()

# Shared Information
def wriggle_SI(q):
    return I_X_Y(q) - cond_I_X_Y__Z(q)
#^ wriggle_SI()


# More Stats stuff

def total_variation_distance(P,Q):
    tvsum = 0.
    for x,p in P.items():
        if x in Q.keys():  tvsum += abs( p - Q[x] )
        else:              tvsum += p
    for x,q in Q.items():
        if x in P.keys():  pass
        else:              tvsum += q
    return tvsum/2.
#^ total_variation_distance()

def support_variation(P,Q): # \sup { P(A) | Q(A)=0 } + vice-vers
    thesum = 0.
    for x,p in P.items():
        if x in Q.keys():  pass
        else:              thesum += p
    for x,q in Q.items():
        if x in P.keys():  pass
        else:              thesum += q
    return thesum/2.
#^ total_variation_distance()

def kl_divergence(Of,From): # KL-divergence of Q from P
    Q=Of
    P=From
    thesum = 0.
    for x,q in Q.items():
        if q>0:
            if x in P.keys():
                p = P[x]
                if    p>0:  thesum += q*log(q/p)
                else:       thesum = 1.e+400
            else:
                thesum = 1.e+400
    return thesum;
#^ kl_divergence()

def sorted_pdf(p):
    p_str = "{"
    lead_str = ""
    for xyz,t in sorted(p.items(), key=lambda i: i[0]):
        p_str += lead_str+str(xyz)+":"+str(t)
        lead_str = ", "
    p_str += "}"
    return p_str
#^ sorted_pdf()

def gradient(p):
    grad = dict()
    p_yz = marginal_yz(p)
    for xyz,t in p.items():
        x,y,z = xyz
        if    p[xyz] > 0:     grad[xyz] = log(p_yz[y,z] / p[x,y,z])
        elif  p_yz[y,z] > 0:  grad[xyz] = 1.e400
        else:                 grad[xyz] = 0.
    return grad
#^ gradient()

def print_solution_stats(soliter, the_p, pdf, true_pdf, true_result, true_CI, true_SI, feas=None, kkt=None):
    if true_pdf    != None: print("SOLUTION #"+str(soliter)+": d_TV(timeseries, true): ", total_variation_distance(pdf, true_pdf))
    if true_result != None: print("SOLUTION #"+str(soliter)+": d_TV(result, true):     ", total_variation_distance(true_result,the_p))
    if true_pdf    != None: print("SOLUTION #"+str(soliter)+": supp-dist(timeseries, true): ", support_variation(pdf, true_pdf))
    if true_result != None: print("SOLUTION #"+str(soliter)+": supp-dist(result, true):     ", support_variation(true_result,the_p))
    if true_pdf    != None: print("SOLUTION #"+str(soliter)+": KL-Divergence(true, timeseries): ", kl_divergence(true_pdf,pdf))
    if true_result != None: print("SOLUTION #"+str(soliter)+": KL-Divergence(true, result):     ", kl_divergence(true_result,the_p))
    print(                        "SOLUTION #"+str(soliter)+": CI-wriggle:",wriggle_CI(pdf,the_p))
    if true_CI     != None: print("SOLUTION #"+str(soliter)+": CI-wriggle difference:",abs( wriggle_CI(pdf,the_p) - true_CI ))
    print(                        "SOLUTION #"+str(soliter)+": SI-wriggle:",wriggle_SI(the_p))
    if true_SI     != None: print("SOLUTION #"+str(soliter)+": SI-wriggle difference:",abs( wriggle_SI(the_p) - true_SI ))
    print(                        "SOLUTION #"+str(soliter)+": 1-norm of violation: ",feas)
    print(                        "SOLUTION #"+str(soliter)+": KKT-system infeasibility: ",kkt)
#^ print_solution_stats()

def solve_PDF(pdf, true_pdf=None, true_result=None, true_CI=None, true_SI=None,
              feas_eps   =1.e-10, kkt_eps   =1.e-5,
              feas_eps_2 =1.e-6,  kkt_eps_2 =.01,
              kkt_search_eps =.5, max_zero_probability=1.e-5, #kkt_search_entropy_eps=1.e-5,
              verbose=False):
    if verbose: solvers.options['show_progress'] = True  # turns the output to the screen on or off
    else:       solvers.options['show_progress'] = False

    # Create marginals
    p_xy = marginal_xy(pdf)
    p_xz = marginal_xz(pdf)
    if verbose: print("solve_PDF(): xy-marginals =",sorted_pdf(p_xy))
    if verbose: print("solve_PDF(): xz-marginals =",sorted_pdf(p_xz))

    # M A I N   L O O P
    wCI=wSI=-1.e+1000
    the_p = None

    feas=1.e+1000
    kkt =1.e+1000
    set_to_zero_list = []
    for main_loop_iter in range(10):
        solvers.options['refinement'] = 2     # number of iterative refinement steps when solving KKT equations (default: 1).
        # create Unique Information solver object
        cui = Compute_UI(p_xy,p_xz, _set_to_zero=set(set_to_zero_list))
        # Let's go!
        cui.solve_it(_eps=1.e-400)
        # Done!
        if verbose: print("solve_PDF(): p_final =",sorted_pdf(cui.p_final))
        if true_result!=None and verbose: print("solve_PDF(): true_result:",sorted(true_result.items(), key=lambda i: i[0]))
        the_p = cui.p_final
        wCI = wriggle_CI(pdf,the_p)
        wSI = wriggle_SI(the_p)

        if verbose: print("solve_PDF(): Testing solution.")
        feas,kkt = cui.check_guess(cui.p_final, kkt_filename="")
        if verbose: print_solution_stats(str(main_loop_iter)+"a", the_p, pdf, true_pdf, true_result, true_CI, true_SI, feas,kkt)
        if kkt<=kkt_eps and feas<=feas_eps:
            if verbose: print("solve_PDF(): Done. Returning a solution with 1-norm of primal violation",feas,"and maximum dual violation ",kkt)
            return the_p,feas,kkt,wCI,wSI
        else:
            if verbose: print("solve_PDF(): Improving solution")
            # zero_probability = entropy_search(cui.p_final,kkt_search_entropy_eps)
            if verbose: print("solve_PDF(): Max ZERO probability = ", max_zero_probability)
            stop_search_kkt_eps = kkt_search_eps*kkt
            kkt_search = cui.search_for_kkt_solution( cui.p_final, show_progress=verbose, stop_search_kkt_eps=stop_search_kkt_eps, stop_search_prob_eps=max_zero_probability)
            if kkt_search==None:
                if verbose: print("solve_PDF(): KKT search turned up empty.")
                if feas<=feas_eps_2 and kkt<=kkt_eps_2:
                    if verbose: print("solve_PDF(): Previous solution seems to be good enough (feas_eps_2, kkt_eps_2) ")
                    if verbose: print("solve_PDF(): Returning a solution with 1-norm of primal violation",feas,"and maximum dual violation ",kkt)
                    return the_p,feas,kkt,wCI,wSI
                else:
                    if verbose: print("solve_PDF(): Giving up.")
                    break   # give up                               B R E A K
            else:
                min_viol,ZERO_min_viol,non_monot,max_viol,kvl = kkt_search
                if ZERO_min_viol==0:
                    if verbose: print("solve_PDF(): KKT search cannot find improvement.")
                    if feas<=feas_eps_2 and kkt<=kkt_eps_2:
                        if verbose: print("solve_PDF(): Previous solution seems to be good enough (feas_eps_2, kkt_eps_2) ")
                        if verbose: print("solve_PDF(): Returning a solution with 1-norm of primal violation",feas,"and maximum dual violation ",kkt)
                        return the_p,feas,kkt,wCI,wSI
                    else:
                        if verbose: print("solve_PDF(): Giving up.")
                        break   # give up                           B R E A K

                if verbose: print("solve_PDF(): Set everything leq",ZERO_min_viol,"to zero.")
                p_new = dict()
                for xyz,t in cui.p_final.items():
                    if t>ZERO_min_viol: p_new[xyz]=t
                feas_new,kkt_new = cui.check_guess(p_new, kkt_filename="")
                if verbose: print("solve_PDF(): Tidied-up solution: ",sorted_pdf(p_new))
                if verbose: print_solution_stats(str(main_loop_iter)+"b",p_new, pdf, true_pdf, true_result, true_CI, true_SI, feas_new,kkt_new)

                for xyz,t in cui.p_final.items():
                    if t <= ZERO_min_viol:   set_to_zero_list.append( xyz )

            del cui
        #^ else (don't `until')
    #^ for iterations (REPEAT-UNTIL)
    print("Giving up.  Returning a solution with 1-norm of primal violation",feas,"and maximum dual violation ",kkt)
    return the_p,feas,kkt,wCI,wSI
# solve_PDF()

#^ EOF TartuSynergy.py
