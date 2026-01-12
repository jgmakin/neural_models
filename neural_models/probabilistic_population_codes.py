# standard libraries
import os
import pdb

# third-party libraries
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# local packages
from utils_jgm.toolbox import rescale, auto_attribute, tau, print_fixed
from utils_jgm.kinematic_chains import NumPyKinematicChains, TorchKinematicChains
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
MCUs = MachineCompatibilityUtils()

rng = default_rng()

# torch if it's there
try:
    import torch
    from torch.utils.data import IterableDataset
    from torch.utils.tensorboard import SummaryWriter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # for tensorboard
    writer = SummaryWriter(
        log_dir=os.path.join(MCUs.get_path('data'), 'MIvDE', 'tf_summaries')
    )

except ModuleNotFoundError:
    print('WARNING: torch missing; skipping')


class ProbabilisticPopulationCodes:

    @auto_attribute
    def __init__(
        self,
        distribution='Poisson',
        tuning_curve_shape='Gaussian',
        nums_units_per_dimension=[10, 10],
        # this will typically be wrong
        stimulus_limits=[[0., 0.], [1., 1.]],
        full_width_at_half_max=1/6,
        # Deneve01: "45-75 degrees"
        name='joint angles',        
        WRAP=False
    ):

        # construct tuning-curve parameters for a [0, 1] x [0, 1] x ... grid
        # Essentially, one std of the tuning curve (in one direction only) is
        #  ~7% of the total grid; so the 1-std (in both directions) coverage
        #  of each tuning curve is 14% of the total area.
        std_dev = self.full_width_at_half_max/(2*( 2*np.log(2) )**(1/2) )
        num_dims = len(nums_units_per_dimension)
        self.tuning_curve_covariance = np.eye(num_dims)*std_dev**2
        self.tuning_curve_precision = np.linalg.inv(
            self.tuning_curve_covariance
        ).astype(np.float32)

        # for stimuli at the margin, only 0.01% will fall off the edge
        self.margin = 0 if WRAP else 4*std_dev

        # ...
        grid_size = 1.0 + 2*self.margin
        preferred_directions = [
            np.linspace(0, grid_size, num_units)
            for num_units in self.nums_units_per_dimension
        ]
        meshes = np.meshgrid(*preferred_directions)
        # pre-construct the preferred directions, [num_dims x num_units]
        self.lattice_PDs = np.array(
            [np.reshape(mesh, -1) for mesh in meshes]
        ).astype(np.float32)

        # init
        self.response = None

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, response):
        self._response = response

    # this will need to get overwritten for Torch or TensorFlow
    def moments_to_samples(self, moments, distribution):
        match distribution:
            case 'Poisson':
                return rng.poisson(moments)
            case 'Bernoulli':
                U = rng.uniform(size=moments.shape)
                return U > moments
            case 'categorical':
                # obviously this should be replaced with a vectorized fxn
                samples = np.zeros(moments.shape[0])
                for iSample, p in enumerate(moments):
                    samples[iSample] = rng.choice(len(p), p=p)

                return samples
            case _:
                raise NotImplementedError(
                    'sampler %s not impl. -- jgm' % distribution
                )

    def stimuli_to_moments(self, stimuli, gains):
        raise NotImplementedError('shell method -- jgm')

    def samples_to_suff_stats(self, samples):
        '''
        samples are expected to be 
            [num_trajectories x T x units1 x units2 x ...],
        or 
            [num_examples x units1 x units2 x ...],
        '''

        # Ns
        num_dims = len(self.nums_units_per_dimension)
        original_shape = samples.shape

        # collapse across dimensions (if it's multidimensional)
        samples = samples.reshape([*samples.shape[:-num_dims], -1])

        # collapse across time (if it's a sequence)
        samples = samples.reshape([-1, samples.shape[-1]])

        match self.tuning_curve_shape:
            case 'Gaussian':

                # get total spike counts, noting where 0
                total_spike_counts = samples.sum(axis=1, keepdims=True)
                bad_inds = (total_spike_counts == 0).nonzero()
                good_inds = (total_spike_counts > 0).nonzero()

                # pytorch's nonzero returns a tensor instead of a tuple
                if not (type(good_inds) is tuple):
                    good_inds = good_inds.T.unbind(dim=0)
                    bad_inds = bad_inds.T.unbind(dim=0)

                ####
                # deal with torus stimuli here
                ####

                # centers of mass
                xhat = samples@self.lattice_PDs.T/total_spike_counts

                # replace bad CoM with random good ones
                if len(bad_inds[0]):
                    print('bad center(s) of mass--randomly assigning estimates!')
                    good_ind_inds = self.moments_to_samples(
                        [[1/len(good_inds[0])]*len(good_inds[0])]*len(bad_inds[0]),
                        'categorical'
                    )
                    xhat[bad_inds[0]] = xhat[good_inds[0][good_ind_inds]]

                # scale back from [0, 1] space into the data space
                xhat = self.grid2world(xhat)

                # restore dimensions to
                #     [num_trajectories x T x num_dims],
                # or 
                #     [num_examples x num_dims],
                return (
                    xhat.reshape([*original_shape[:-num_dims], num_dims]),
                    total_spike_counts
                )

            case _:
                raise NotImplementedError('Non-Gaussian tuning! -- jgm')

    @property
    def posterior_parameters(self):
        raise NotImplementedError('shell method -- jgm')

    def grid2world(self, xhat):
        raise NotImplementedError('shell method -- jgm')

    def grid_shape(self, data_matrix):
        N_cases = data_matrix.shape[0]
        return data_matrix.reshape([N_cases] + self.nums_units_per_dimension)

    def respond(self, stimuli, gains):
        moments = self.stimuli_to_moments(stimuli, gains)
        samples = self.moments_to_samples(moments, self.distribution)
        self.response = samples.reshape([-1, *self.nums_units_per_dimension])


class NumPyPPCs(ProbabilisticPopulationCodes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stimulus_limits = np.array(self.stimulus_limits)
    
    def stimuli_to_moments(self, stimuli, gains):
        match self.tuning_curve_shape:
            case 'Gaussian':
                # rescale into the units of a [0, 1] x [0, 1] x ... grid
                stimuli = rescale(
                    stimuli,
                    self.stimulus_limits[0, :],
                    self.stimulus_limits[1, :],
                    np.array([self.margin]*len(self.nums_units_per_dimension)),
                    np.array([self.margin + 1.0]*len(self.nums_units_per_dimension))
                )

                # map into tuning-curve response (the means/moments)
                Info = self.tuning_curve_precision
                InfoXi = Info@self.lattice_PDs
                quadratic_term = -(stimuli@Info*stimuli).sum(1, keepdims=True)/2
                linear_term = stimuli@InfoXi
                constant_term = -(self.lattice_PDs*InfoXi).sum(0, keepdims=True)/2
                quadratic_form = quadratic_term + linear_term + constant_term

                # and now scale by gains and exponentiate
                #print(linear_term.shape, constant_term.shape, quadratic_term.shape)
                return gains*np.exp(quadratic_form)  #, stimuli

            case _:
                raise NotImplementedError(
                    '%s tuning curves not implemented -- jgm' % self.tuning_curve_shape
                )

    def grid2world(self, gridEstimates):
        return rescale(
            gridEstimates,
            np.array([self.margin]*len(self.nums_units_per_dimension)),
            np.array([self.margin + 1.0]*len(self.nums_units_per_dimension)),
            ###################
            # is this brittle?
            self.stimulus_limits[0, :], self.stimulus_limits[1, :],
            ###################
        )

    @property
    def posterior_parameters(self):
        xhat, total_spike_counts = self.samples_to_suff_stats(self.response)
        stimulus_range = np.diagflat(np.diff(self.stimulus_limits, axis=0))

        tuning_precision = stimulus_range@self.tuning_curve_precision@stimulus_range
        stimulus_precision = tuning_precision[None, :, :]*total_spike_counts[:, :, None]

        return {
            'mean': xhat,
            'precision': stimulus_precision,
        }        
        

class TorchPPCs(ProbabilisticPopulationCodes):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # put on GPU
        self.tuning_curve_covariance = torch.tensor(self.tuning_curve_covariance).to(device)
        self.tuning_curve_precision = torch.tensor(self.tuning_curve_precision).to(device)
        self.margin = torch.tensor(self.margin).to(device)
        self.lattice_PDs = torch.tensor(self.lattice_PDs).to(device)
        self.stimulus_limits = torch.tensor(self.stimulus_limits).to(device)

    def moments_to_samples(self, moments, distribution):
        match distribution:
            case 'Poisson':
                return torch.distributions.Poisson(moments).sample()
            case 'Bernoulli':
                return torch.distributions.Bernoulli(moments).sample()
            case 'categorical':
                return torch.distributions.Categorical(
                    torch.as_tensor(moments)).sample()
            case _:
                raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def stimuli_to_moments(self, stimuli, gains):    
        match self.tuning_curve_shape:
            case 'Gaussian':
                # rescale into the units of a [0, 1] x [0, 1] x ... grid
                stimuli = rescale(
                    stimuli,
                    self.stimulus_limits[0, :],
                    self.stimulus_limits[1, :],
                    torch.full([len(self.nums_units_per_dimension)], self.margin),
                    torch.full([len(self.nums_units_per_dimension)], self.margin + 1.0),
                )

                # map into tuning-curve response (the means/moments)
                Info = self.tuning_curve_precision
                InfoXi = Info@self.lattice_PDs
                quadratic_term = -(stimuli@Info*stimuli).sum(1, keepdims=True)/2
                linear_term = stimuli@InfoXi
                constant_term = -(self.lattice_PDs*InfoXi).sum(0, keepdims=True)/2
                quadratic_form = quadratic_term + linear_term + constant_term

                # and now scale by gains and exponentiate
                #print(linear_term.shape, constant_term.shape, quadratic_term.shape)
                return gains*torch.exp(quadratic_form)  #, stimuli

            case _:
                raise NotImplementedError(
                    '%s tuning curves not implemented -- jgm' % self.tuning_curve_shape
                )
    
    def grid2world(self, gridEstimates):
        return rescale(
            gridEstimates,
            torch.full([len(self.nums_units_per_dimension)], self.margin),
            torch.full([len(self.nums_units_per_dimension)], self.margin + 1.0),
            ###################
            # is this brittle?
            self.stimulus_limits[0, :], self.stimulus_limits[1, :],
            ###################
        )
    
    @property
    def posterior_parameters(self):
        # extract the sufficient stats from this object's response (samples)
        xhat, total_spike_counts = self.samples_to_suff_stats(self.response)

        # convert precision into stimulus space, and scale by total_spike_counts
        inverse_stim_range = torch.diagflat(1/torch.diff(self.stimulus_limits, axis=0))
        tuning_precision = inverse_stim_range@self.tuning_curve_precision@inverse_stim_range
        stimulus_precision = tuning_precision[None, :, :]*total_spike_counts[:, :, None]

        # the posterior is Gaussian with these parameters
        return {
            'mean': xhat,
            'precision': stimulus_precision,
        }        
        

class MultisensoryData:
    def __init__(
        self,
        nums_units_per_dimension=[30, 30],
        gain_limits=[12., 18.]
    ):

        # 
        self.gain_limits = gain_limits

        # create a robot arm
        self.robot = self.KinematicChains()
    
        # two populations, proprioceptive and visual
        self.PPCs = {
            'hand position': self.PPC_class(
                stimulus_limits=self.robot.position_limits,
                nums_units_per_dimension=nums_units_per_dimension,
                name='hand position'
            ),
            'joint angles': self.PPC_class(
                stimulus_limits=self.robot.joint_limits,
                nums_units_per_dimension=nums_units_per_dimension,
                name='joint angles',
            ),
        }
        self.gains = {
            'joint angles': None,
            'hand position': None,
        }

    @property
    def KinematicChains(self):
        raise NotImplementedError('shell method -- jgm')

    @property
    def PPC_class(self):
        raise NotImplementedError('shell method -- jgm')

    def generate_latents(self, num_examples=4000):
        raise NotImplementedError('shell method -- jgm')

    def generate_patents(self):

        if None in self.gains.values():
            raise ValueError('You have to run generate_latents first')

        self.PPCs['joint angles'].respond(
            stimuli=self.robot.joint_angles, gains=self.gains['joint angles']
        )
        self.PPCs['hand position'].respond(
            stimuli=self.robot.position, gains=self.gains['hand position']
        )

    def get_observed_data(self, num_examples=4000, FLAT=True):
        '''
        Generate num_examples latent and observed data and return the latter.
        The responses of the two populations may either be flattened and
        concatenated (FLAT=True) or (if the populations have the same size and
        shape) stacked along a new dimension (dim=1).
        '''

        self.generate_latents(num_examples=num_examples)
        self.generate_patents()
        return self.join_populations([PPC.response for PPC in self.PPCs.values()], FLAT)


class NumPyMultisensoryData(MultisensoryData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.joint_limits = np.array([joint.limits for joint in self.robot.joints])
        
    @property
    def KinematicChains(self):
        return NumPyKinematicChains

    @property
    def PPC_class(self):
        return NumPyPPCs

    def generate_latents(self, num_examples=4000):
        num_joints = self.PPCs['joint angles'].tuning_curve_covariance.shape[0]

        # generate random robot configurations, uniform in joint-angle space
        self.robot.joint_angles = rescale(
            rng.uniform(size=(num_examples, num_joints)),
            np.zeros([1, 2]), np.ones([1, 2]),
            self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        # generate random population gains
        self.gains['hand position'] = rescale(
            rng.uniform(size=(num_examples, 1)),
            np.array(0), np.array(1),
            np.array(self.gain_limits[0]), np.array(self.gain_limits[1])
        )

        self.gains['joint angle'] = rescale(
            rng.uniform(size=(num_examples, 1)),
            np.array(0), np.array(1),
            np.array(self.gain_limits[0]), np.array(self.gain_limits[1])
        )

    @staticmethod
    def join_populations(response_list, FLAT=True):
        '''
        The inverse of `split_populations`
        '''

        if FLAT:
            # concatentate flattened populations
            # It's tempting to stack the populations and then flatten, for
            #  consistency with the FLAT=False case below.  But that rules out
            #  having populations of different shapes and sizes.
            return np.concatenate((
                R.reshape([R.shape[0], -1]) for R in response_list
            ), axis=1)
        else:
            # stack population responses along a new dimension (dim=1)
            # I.e., treat each population as a "channel," i.e. an N x N array.
            #  This assumes that the population have the same shape!
            return np.stack(response_list, axis=1)

    @staticmethod
    def split_populations(data, nums_units_per_pop, FLAT=True):
        '''
        The inverse of `join_populations`

        NB: if FLAT, a list is returned, else a tensor; but in either case,
            enumerating over the first dimension (0) will extract pops.
        '''

        if FLAT:
            split_indices = np.cumsum(nums_units_per_pop)[:-1]
            return np.split(data, split_indices, axis=1)
        else:
            return data.permute([1, 0, 2, 3])
    

class TorchMultisensoryData(MultisensoryData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.joint_limits = torch.vstack([
        #     torch.tensor(joint.limits) for joint in self.robot.joints
        # ])
        self.joint_limits = torch.tensor([joint.limits for joint in self.robot.joints])
        
    @property
    def KinematicChains(self):
        return TorchKinematicChains

    @property
    def PPC_class(self):
        return TorchPPCs

    def generate_latents(self, num_examples=4000):
        num_joints = self.PPCs['joint angles'].tuning_curve_covariance.shape[0]

        # generate random robot configurations, uniform in joint-angle space
        self.robot.joint_angles = rescale(
            torch.distributions.Uniform(0, 1).sample((num_examples, num_joints)),
            torch.zeros([1, 2]), torch.ones([1, 2]),
            self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        # generate random population gains
        self.gains['hand position'] = rescale(
            torch.distributions.Uniform(0, 1).sample((num_examples, 1)),
            torch.tensor(0), torch.tensor(1),
            torch.tensor(self.gain_limits[0]), torch.tensor(self.gain_limits[1])
        )
        self.gains['joint angles'] = rescale(
            torch.distributions.Uniform(0, 1).sample((num_examples, 1)),
            torch.tensor(0), torch.tensor(1),
            torch.tensor(self.gain_limits[0]), torch.tensor(self.gain_limits[1])
        )

    @staticmethod
    def join_populations(response_list, FLAT=True):
        '''
        The inverse of `split_populations`
        '''

        if FLAT:
            # concatentate flattened populations
            # It's tempting to stack the populations and then flatten, for
            #  consistency with the FLAT=False case below.  But that rules out
            #  having populations of different shapes and sizes.
            return torch.concatenate(tuple(
                R.reshape([R.shape[0], -1]) for R in response_list
            ), axis=1)
        else:
            # stack population responses along a new dimension (dim=1)
            # I.e., treat each population as a "channel," i.e. an N x N array.
            #  This assumes that the population have the same shape!
            return torch.stack(response_list, axis=1)

    @staticmethod
    def split_populations(data, nums_units_per_pop, FLAT=True):
        '''
        The inverse of `join_populations`

        NB: if FLAT, a list is returned, else a tensor; but in either case,
            enumerating over the first dimension (0) will extract pops.
        '''

        if FLAT:
            return torch.split(data, nums_units_per_pop, dim=1)
        else:
            return data.permute([1, 0, 2, 3])
        

class MultisensoryDataset(IterableDataset):
    def __init__(self, N_examples_per_batch, h, w, CONV=False):
        super().__init__()

        self.N_examples_per_batch = N_examples_per_batch
        self.multisensory_data = TorchMultisensoryData(
            nums_units_per_dimension=[h, w],
        )
        self.CONV = CONV

    def __iter__(self):
        # This loop makes it infinite
        while True:
            V0 = self.multisensory_data.get_observed_data(
                self.N_examples_per_batch, FLAT=(not self.CONV)
            )
            label = None

            yield V0, label


class EFHtester():
    def __init__(self, N_examples_per_batch, h, w, CONV=False):
        self.N_examples_per_batch = N_examples_per_batch
        self.multisensory_data = TorchMultisensoryData(
            nums_units_per_dimension=[h, w],
        )
        self.nums_units_per_pop = [
            np.prod(PPC.nums_units_per_dimension)
            for PPC in self.multisensory_data.PPCs.values()
        ]
        self.CONV = CONV

    @torch.no_grad()
    def __call__(self, efh, step, VERBOSE=False):
        FLAT = not self.CONV
        Y0 = self.multisensory_data.get_observed_data(
            num_examples=self.N_examples_per_batch, FLAT=FLAT
        )

        # deterministic up-down through model
        mu_hid, _ = efh.infer(Y0)
        Y1, _ = efh.emit(mu_hid)
        Y1_split = self.multisensory_data.split_populations(
            Y1, self.nums_units_per_pop, FLAT=FLAT
        )

        # test
        X = self.multisensory_data.robot.joint_angles
        J = self.multisensory_data.robot.forward_kinematics_Jacobian(X)

        # init
        N_examples, N_dims = X.shape
        XhatOpt = torch.zeros((X.shape))
        precisionsOpt = torch.zeros([*X.shape, X.shape[-1]])
        error_lengths = {}

        # loop over both *updated* PPCs
        for PPC, R1 in zip(self.multisensory_data.PPCs.values(), Y1_split):
            params = PPC.posterior_parameters
            Xhat0 = params['mean']
            precision = params['precision']
            R1 = PPC.grid_shape(R1)  # does nothing if already in grid_shape
            Xhat1, _ = PPC.samples_to_suff_stats(R1)

            if PPC.name == 'hand position':

                # convert mean to joint-angles space
                # X = PPCs.robot.inverse_kinematics(X)   # not necessary....
                Xhat0 = self.multisensory_data.robot.inverse_kinematics(Xhat0)
                Xhat1 = self.multisensory_data.robot.inverse_kinematics(Xhat1)

                # convert precision to joint-angles space
                precision = torch.einsum(
                    'nmj, njk, nkl -> nml', J.mT, precision, J
                )

            # numerical errors sometimes cause negative det, so use nanmean
            if VERBOSE:
                avg_cov_size = (1/precision.det()**(1/N_dims)).nanmean(0)
                print_fixed(PPC.name + ' covariance', avg_cov_size)

            # compute average "error size," a la normalizer for Gaussian
            for Xhat, name in zip([Xhat0, Xhat1], [PPC.name, PPC.name + ' updated']):
                error_lengths[name] = errors_to_error_size(X - Xhat)

            XhatOpt += torch.einsum('nml, nl -> nm', precision, Xhat0)
            precisionsOpt += precision

        if VERBOSE:
            print_fixed('optimal precision', (1/precisionsOpt.det()**(1/N_dims)).mean(0))

        XhatOpt = torch.linalg.solve(precisionsOpt, XhatOpt[:, :, None]).squeeze(-1)
        error_lengths['optimal'] = errors_to_error_size(X - XhatOpt)

        # print
        print_fixed(
            "integ error", error_lengths['joint angles updated'], 10, 2, -6,
            end='',
        )

        # write for tensorboard
        writer.add_scalars(
            'integration',
            {
                'angle error (input)': error_lengths['joint angles'],
                'angle error (EFH)': error_lengths['joint angles updated'],
                'angle error (opt)': error_lengths['optimal'],
            },
            step
        )

        return error_lengths

    @torch.no_grad()
    def plot_updated(self, efh):
        FLAT = not self.CONV

        Y0 = self.multisensory_data.get_observed_data(1, FLAT)
        mu_hid, _ = efh.infer(Y0)
        Y1, _ = efh.emit(mu_hid)

        # cols=2 because we need a col for Y0 and a col for Y1
        fig, AXES = plt.subplots(len(self.multisensory_data.PPCs), 2)
        for iGibbs, Y in enumerate([Y0, Y1]):
            Y_list = self.multisensory_data.split_populations(
                Y, self.nums_units_per_pop, FLAT=FLAT
            )
            for iPop, (PPC, response) in enumerate(zip(
                self.multisensory_data.PPCs.values(), Y_list
            )):
                # does nothing if already in grid_shape
                response = PPC.grid_shape(response)
                AXES[iPop, iGibbs].imshow(response[0].cpu().detach())

    @torch.no_grad()
    def plot_generated(self, efh, N_CD_steps, H, W, num_examples):
        FLAT = not self.CONV

        if self.CONV:
            Yhats = efh.generate(N_CD_steps, num_examples, H, W)
        else:
            Yhats = efh.generate(N_CD_steps, num_examples)

        Yhat_list = self.multisensory_data.split_populations(
            Yhats, self.nums_units_per_pop, FLAT=FLAT
        )

        # a row for each population
        fig, AXES = plt.subplots(2, num_examples)
        for iPop, (PPC, Yhat) in enumerate(zip(
            self.multisensory_data.PPCs.values(), Yhat_list
        )):
            # does nothing if already in grid_shape
            Yhat = PPC.grid_shape(Yhat)
            for iExample, yhat in enumerate(Yhat):
                AXES[iPop, iExample].imshow(yhat.cpu().detach())


# helper functions
def errors_to_error_size(E):
    N_examples, N_dims = E.shape
    return ((E.T@E)).det()**(1/N_dims)/N_examples


class LTIPPCs:
    def __init__(
        self,
        nums_units_per_dimension=[15],
        gain_limits=[6.4, 9.6],
        mass=5,
        stiffness=3,
        damping=0.25,
        dt=0.05,
        xmax=[tau/6, 0.8],
        xmin=[-tau/6, -0.8],
    ):

        # gains
        self.gain_limits = gain_limits
        self.gains = None

        # Ns
        num_dims = len(nums_units_per_dimension)
        num_states = 2*num_dims  # second-order dynamics

        # state-space parameters
        m = mass; k = stiffness; b = damping
        self.A = np.block([
            [np.eye(num_dims), dt*np.eye(num_dims)],
            [-k/m*dt*np.eye(num_dims), -(b/m*dt-1)*np.eye(num_dims)],
        ])
        # we observe only positions
        self.C = np.zeros([num_dims, num_states])
        self.C[:num_dims, :num_dims] = np.eye(num_dims)

        # transition noise
        variance_pos = 5e-7
        variance_vel = 5e-5
        self.transition_covariance = np.block([
            [variance_pos*np.eye(num_dims), np.zeros(num_dims)],
            [np.zeros(num_dims), variance_vel*np.eye(num_dims)]
        ])

        # prior distribution---assumes no position-velocity coupling
        xmin = np.array(xmin); xmax = np.array(xmax)
        self.position0_mean         = ((xmax-xmin)/2 + xmin)[:num_dims];
        self.position0_covariance   = np.eye(num_dims)*np.inf  ####
        self.velocity0_mean         = np.zeros([num_dims, 1]);
        self.velocity0_covariance   = np.eye(num_dims)*5e-10;
        self.xmin = xmin
        self.xmax = xmax

        # probabilistic population code
        self.position_PPC = ProbabilisticPopulationCodes(
            stimulus_limits=np.block([[xmin[:num_dims]], [xmax[:num_dims]]]),
            nums_units_per_dimension=nums_units_per_dimension, WRAP=True
        )

    def generate_latents(self, num_trajectories, T=4000):

        # initialize
        num_states = self.A.shape[0]
        state = np.zeros([T, num_states, num_trajectories])
        pos0 = uniform_normal_dirac_sampler(
            num_trajectories, self.position0_mean, self.position0_covariance,
            self.xmin[:num_states//2], self.xmax[:num_states//2], 0.05
        )
        vel0 = uniform_normal_dirac_sampler(
            num_trajectories, self.velocity0_mean, self.velocity0_covariance,
            self.xmin[-num_states//2:], self.xmax[-num_states//2:], 0.05
        )
        state[0, :, :] = np.block([[pos0], [vel0]])

        # pre-compute transition noise
        transition_std = np.linalg.cholesky(self.transition_covariance)
        transition_noise = np.transpose(np.reshape(
            transition_std@rng.normal(size=[num_states, num_trajectories*T]),
            [num_states, num_trajectories, T]
        ), (2, 0, 1))

        # iterate
        for t in range(T-1):
            state[t+1, :, :] = self.A@state[t, :, :] + transition_noise[t, :, :]
        self.latent_state = np.transpose(state, [2, 0, 1])

        # generate random population gains
        num_populations = 1  ### hard coded
        self.gains = rescale(
            rng.uniform(size=(num_trajectories, T, num_populations)),
            0, 1, self.gain_limits[0], self.gain_limits[1]
        )

    def generate_patents(self):

        if self.gains is None:
            raise ValueError('You have to run generate_latents first')

        # Ns
        num_obsvs = self.C.shape[0]
        num_trajectories, T, num_states = self.latent_state.shape
        num_populations = 1  ### hard-coded
        
        # response of probabilistic population codes
        stimuli = self.latent_state@self.C.T
        vis = self.position_PPC.stimuli_to_samples(
            stimuli=stimuli.reshape([-1, num_obsvs]),
            gains=self.gains.reshape([num_trajectories*T, num_populations])
        )

        # separate out trajectories
        return vis.reshape([num_trajectories, T, -1])

    def get_observed_data(self, num_trajectories=10, T=4000):
        '''
        For use with training models.  Data have shape

        '''

        self.generate_latents(num_trajectories, T)
        vis = self.generate_patents()

        return vis


def uniform_normal_dirac_sampler(num_samples, mu, Sigma, xmin, xmax, margin):
    '''
    NB: X has size (num_dims x num_samples)
    '''

    num_dims = len(mu)

    # ...
    if Sigma.sum() == np.inf:
        # infinite covariance; sample uniformly
        X = rescale(
            rng.uniform(size=(num_samples, num_dims)),
            [0]*num_dims, [1]*num_dims, xmin + margin, xmax - margin
        ).T
    elif Sigma.sum() == 0:
        # infinite precision; sample from a Dirac delta
        X = np.tile(np.reshape(mu, [-1, 1]), [1, num_samples])
    else:    
        # sample from a normal distribution
        X = mu + np.linalg.cholesky(Sigma)@rng.normal(
            size=[num_dims, num_samples]
        )
    return X


def wrap_stimuli(stimuli_not_wrapped, stimulus_minima, stimulus_maxima, N):
    '''
    Wrap stimuli onto an M-torus
    '''

    #####
    # This is complicated.  See
    #   wrapStimuli.m, getDataPPC.m, getBestToroidalEsts.m
    #####

    pass
