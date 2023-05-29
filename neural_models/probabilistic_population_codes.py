import pdb
import numpy as np
from numpy.random import default_rng

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    import tfmpl
    import machine_learning.neural_networks.tf_helpers as tfh
    from utils_jgm.kinematic_chains import TensorFlowKinematicChains
except ModuleNotFoundError as e:
    print('WARNING:', end=' ')
    print(e)
    print('proceeding without it...')

from utils_jgm.toolbox import rescale, auto_attribute, tau
from utils_jgm.kinematic_chains import NumPyKinematicChains

rng = default_rng()

inverse_link_dict = {
    'Poisson': np.exp,
    'Bernoulli': lambda Eta: 1/(1 + np.exp(-Eta)),
    'StandardNormal': lambda Eta: Eta,
}


class ProbabilisticPopulationCodes:

    @auto_attribute
    def __init__(
        self,
        distribution='Poisson',
        tuning_curve_shape='Gaussian',
        nums_units_per_dimension=[10, 10],
        # this will typically be wrong
        stimulus_limits=[[0, 0], [1, 1]],
        full_width_at_half_max=1/6,
        # Deneve01: "45-75 degrees"
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
        )

        # for stimuli at the margin, only 0.01% will fall off the edge
        self.margin = 0 if WRAP else 4*std_dev

    def stimuli_to_moments(self, stimuli, gains):

        '''
        stimuli: Ndims x Nexamples
        '''
        raise NotImplementedError('shell method -- jgm')

    def stimuli_to_samples(self, stimuli, gains):
        moments = self.stimuli_to_moments(stimuli, gains)
        return self.moments_to_samples(moments, self.distribution)

    def moments_to_samples(self, moments, distribution):
        raise NotImplementedError('shell method -- jgm')

    def grid_shape(self, data_matrix):
        raise NotImplementedError('shell method -- jgm')


class NumPyPPCs(ProbabilisticPopulationCodes):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # pre-construct the preferred directions, [num_dims x num_units]
        grid_size = 1.0 + 2*self.margin
        preferred_directions = [
            np.linspace(0, grid_size, num_units)
            for num_units in self.nums_units_per_dimension
        ]
        meshes = np.meshgrid(*preferred_directions)
        self.lattice_PDs = np.array(
            [np.reshape(mesh, -1) for mesh in meshes]
        )

    def stimuli_to_moments(self, stimuli, gains):

        if self.tuning_curve_shape == 'Gaussian':
            # rescale into the units of a [0, 1] x [0, 1] x ... grid
            stimuli = rescale(
                stimuli,
                np.array(self.stimulus_limits)[0, :],
                np.array(self.stimulus_limits)[1, :],
                [self.margin]*len(self.nums_units_per_dimension),
                [self.margin + 1.0]*len(self.nums_units_per_dimension)
            )

            # map into tuning-curve response (the means/moments)
            Info = self.tuning_curve_precision
            InfoXi = Info@self.lattice_PDs
            quadratic_term = -np.sum(stimuli@Info*stimuli, 1, keepdims=True)/2
            linear_term = stimuli@InfoXi
            constant_term = -np.sum(self.lattice_PDs*InfoXi, 0, keepdims=True)/2
            quadratic_form = quadratic_term + linear_term + constant_term

            # and now scale by gains and exponentiate
            #print(linear_term.shape, constant_term.shape, quadratic_term.shape)
            return gains*np.exp(quadratic_form) #, stimuli

        else:
            print('Oops: haven''t programmed this case yet')

    def moments_to_samples(self, moments, distribution):
        if distribution == 'Poisson':
            return rng.poisson(moments)
        elif distribution == 'Bernoulli':
            U = rng.uniform(size=moments.shape)
            return U > moments
        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def samples_to_estimates(self, samples):
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

        if self.tuning_curve_shape == 'Gaussian':

            # ...
            total_spike_counts = np.sum(samples, axis=1, keepdims=True)

            ####
            # deal with torus stimuli here
            ####

            # centers of mass
            xhat = samples@self.lattice_PDs.T/total_spike_counts

            # scale back from [0, 1] space into the data space
            xhat = rescale(
                xhat,
                [self.margin]*len(self.nums_units_per_dimension),
                [self.margin + 1.0]*len(self.nums_units_per_dimension),
                ###################
                # is this brittle?
                np.array(self.stimulus_limits)[0, :],
                np.array(self.stimulus_limits)[1, :],
                ###################
            )

            # restore dimensions to
            #     [num_trajectories x T x num_dims],
            # or 
            #     [num_examples x num_dims],
            return xhat.reshape([*original_shape[:-num_dims], num_dims])

        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def grid_shape(self, data_matrix):
        N_cases = data_matrix.shape[0]
        return np.reshape(
            data_matrix, [N_cases] + self.nums_units_per_dimension
        )


class TensorFlowPPCs(ProbabilisticPopulationCodes):

    @auto_attribute
    def __init__(
        self,
        distribution='Poisson',
        tuning_curve_shape='Gaussian',
        nums_units_per_dimension=[10, 10],
        stimulus_limits=[[0, 0], [1, 1]],
        full_width_at_half_max=1/6,
        # Deneve01: "45-75 degrees"
        WRAP=False
    ):

        super().__init__(**kwargs)

        grid_size = 1.0 + 2*self.margin
        preferred_directions = [
            tf.linspace(0.0, grid_size, num_units)
            for num_units in self.nums_units_per_dimension
        ]
        meshes = tf.meshgrid(*preferred_directions)
        self.lattice_PDs = tf.stack(
            [tf.reshape(mesh, [-1]) for mesh in meshes], axis=0)

    def stimuli_to_moments(self, stimuli, gains):

        if self.tuning_curve_shape == 'Gaussian':
            # rescale into the units of a [0, 1] x [0, 1] x ... grid
            stimuli = tfh.rescale(
                stimuli,
                np.array(self.stimulus_limits)[None, 0, :],
                np.array(self.stimulus_limits)[None, 1, :],
                np.array([self.margin]*len(
                    self.nums_units_per_dimension)
                )[None, :],
                np.array([self.margin + 1.0]*len(
                    self.nums_units_per_dimension)
                )[None, :]
            )

            # map into tuning-curve response (the means/moments)
            Sigma = tf.constant(
                self.tuning_curve_precision, dtype=tf.float32
            )
            SigmaXi = Sigma@self.lattice_PDs
            quadratic_term = -tf.reduce_sum(
                stimuli@Sigma*stimuli, axis=1, keepdims=True
            )/2
            linear_term = stimuli@SigmaXi
            constant_term = -tf.reduce_sum(
                self.lattice_PDs*SigmaXi, axis=0, keepdims=True
            )/2
            quadratic_form = quadratic_term + linear_term + constant_term

            # and now scale by gains and exponentiate
            return gains*tf.exp(quadratic_form)

        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def moments_to_samples(moments, distribution):
        if distribution == 'Poisson':
            return tfp.distributions.Poisson(moments).sample()
        elif distribution == 'Bernoulli':
            return tfp.distributions.Bernoulli(moments).sample()
        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def samples_to_estimates(self, samples):

        if self.tuning_curve_shape == 'Gaussian':
            total_spike_counts = tf.reduce_sum(samples, axis=1, keepdims=True)

            ####
            # deal with torus stimuli here
            ####

            # centers of mass
            return samples@tf.transpose(self.lattice_PDs)/total_spike_counts
        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')

    def grid_shape(self, data_matrix):
        # reshape into the grid
        N_cases = tf.shape(data_matrix)[0]
        return tf.reshape(
            data_matrix, [N_cases] + self.nums_units_per_dimension
        )
        

def split_populations(data_matrix, PPCs):
    if type(data_matrix) is np.ndarray:
        split_indices = np.cumsum(
            [np.prod(PPC.nums_units_per_dimension) for PPC in PPCs]
        )[:-1]
        return np.split(data_matrix, split_indices, axis=1)
    elif type(data_matrix) is tf.Tensor:
        split_indices = [np.prod(PPC.nums_units_per_dimension) for PPC in PPCs]
        return tf.split(data_matrix, split_indices, axis=1)
    else:
        raise TypeError('Input stimuli are of unexpected type! -- jgm')


class MultisensoryPPCs:

    def __init__(
        self,
        nums_units_per_dimension=[30, 30],
        gain_limits=[12, 18]
    ):

        # 
        self.gain_limits = gain_limits
        self.gains = None

        # create a robot arm
        self.robot = NumPyKinematicChains()

        # two populations, proprioceptive and visual
        joint_limits = np.array([joint.limits for joint in self.robot.joints])
        self.joint_angles_PPC = NumPyPPCs(
            stimulus_limits=joint_limits,
            nums_units_per_dimension=nums_units_per_dimension,
        )
        self.position_PPC = NumPyPPCs(
            stimulus_limits=[ [-28.4853, -11.5147], [26.1421, 29.7222] ],
            ###
            # hard-coded--fix this! should be calculated automatically in
            #  KinematicChains
            ###
            nums_units_per_dimension=nums_units_per_dimension,
        )

    def generate_latents(self, num_examples=4000):
        num_joints = self.joint_angles_PPC.tuning_curve_covariance.shape[0]

        # generate random robot configurations, uniform in joint-angle space
        joint_limits = np.array([joint.limits for joint in self.robot.joints])
        self.robot.joint_angles = rescale(
            rng.uniform(size=(num_examples, num_joints)),
            np.zeros([1, 2]), np.ones([1, 2]),
            joint_limits[:, 0], joint_limits[:, 1]
        )

        # generate random population gains
        num_populations = 2  # hard coded: one prop, one vis....
        self.gains = rescale(
            rng.uniform(size=(num_examples, num_populations)),
            0, 1, self.gain_limits[0], self.gain_limits[1]
        )

    def generate_patents(self):

        if self.gains is None:
            raise ValueError('You have to run generate_latents first')

        # response of probabilistic population codes
        prop = self.joint_angles_PPC.stimuli_to_samples(
            stimuli=self.robot.joint_angles,
            gains=self.gains[:, 0, None]
        )
        vis = self.position_PPC.stimuli_to_samples(
            stimuli=self.robot.position[:, :2],
            gains=self.gains[:, 1, None]
        )
        return (
            prop.reshape([-1, *self.joint_angles_PPC.nums_units_per_dimension]),
            vis.reshape([-1, *self.position_PPC.nums_units_per_dimension])
        )

    def get_data(self, num_examples=4000):
        '''
        For use with training models (data are flattened and concatenated together)
        '''

        self.generate_latents(num_examples=4000)
        prop, vis = self.generate_patents()

        return np.concatenate(
            (prop.reshape([prop.shape[0], -1]), vis.reshape([vis.shape[0], -1])),
            axis=1
        )


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
        self.position_PPC = NumPyPPCs(
            stimulus_limits=np.block([[xmin[:num_dims]], [xmax[:num_dims]]]),
            nums_units_per_dimension=nums_units_per_dimension,
        )

    def generate_latents(self, num_trajectories, T=4000):

        # initialize
        num_states = self.A.shape[0]
        state = np.zeros([T, num_states, num_trajectories])
        pos0 = UniformNormalDiracSampler(
            num_trajectories, self.position0_mean, self.position0_covariance,
            self.xmin[:num_states//2], self.xmax[:num_states//2], 0.05
        )
        vel0 = UniformNormalDiracSampler(
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
        ######
        # this works??
        stimuli = (self.latent_state@self.C.T).reshape([-1, num_obsvs])
        ######
        vis = self.position_PPC.stimuli_to_samples(
            stimuli=stimuli,
            gains=self.gains.reshape([num_trajectories*T, num_populations])
        )

        # separate out trajectories
        return vis.reshape([num_trajectories, T, -1])

    def get_data(self, num_trajectories=10, T=4000):
        '''
        For use with training models.  Data have shape

        '''

        self.generate_latents(num_trajectories, T)
        vis = self.generate_patents()

        return vis


def UniformNormalDiracSampler(num_samples, mu, Sigma, xmin, xmax, margin):
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







################
# FIX ME
################
def multisensory_integration_data_generator(
    num_joints=2,
    nums_units_per_dimension=[30, 30],
    num_examples=4000,
    num_cases=40,
    gain_limits=[12, 18]
):

    # nums
    num_populations = 2

    # create a robot arm
    robot = TensorFlowKinematicChains()

    # generate random robot configurations
    joint_limits = np.array([joint.limits for joint in robot.joints])
    th = tfh.rescale(
        tf.random_uniform((num_examples, num_joints)),
        np.array([0, 0]), np.array([1, 1]),
        joint_limits[:, 0], joint_limits[:, 1]
    )
    robot.joint_angles = th

    # generate random population gains
    gains = tfh.rescale(
        tf.random_uniform((num_examples, num_populations)),
        0, 1, gain_limits[0], gain_limits[1]
    )

    # two populations, proprioceptive and visual
    joint_angles_PPC = ProbabilisticPopulationCodes(
        stimulus_limits=joint_limits,
        nums_units_per_dimension=nums_units_per_dimension,
    )
    position_PPC = ProbabilisticPopulationCodes(
        stimulus_limits=[ [-28.4853, -11.5147], [26.1421, 29.7222] ],
        ###
        # hard-coded--fix this! should be calculated automatically in
        #  KinematicChains
        ###
        nums_units_per_dimension=nums_units_per_dimension,
    )

    # create tensorflow tensors
    sample_joint_angles_PPC = joint_angles_PPC.stimuli_to_samples(
        stimuli=robot.joint_angles,
        gains=tf.expand_dims(gains[:, 0], axis=1)
    )
    sample_position_PPC = position_PPC.stimuli_to_samples(
        stimuli=robot.position[:, :2],
        gains=tf.expand_dims(gains[:, 1], axis=1)
    )
    sample_PPC = tf.concat(
        (sample_joint_angles_PPC, sample_position_PPC), axis=1)

    # create a tensorflow dataset from these
    dataset = tf.data.Dataset.from_tensor_slices(sample_PPC)
    dataset = dataset.padded_batch(
        num_cases, padded_shapes=dataset.output_shapes)

    return dataset, joint_angles_PPC, position_PPC



# @tfmpl.figure_tensor
# def plot_populations(data_vector, PPCs):
#     '''Thing'''

#     vectors = split_populations(data_vector, PPCs)
#     populations = [PPC.grid_shape(vector) for vector, PPC in zip(vectors, PPCs)]

#     # input_data
#     figs = tfmpl.create_figures(len(populations), figsize=(2, 2))
#     for fig, population in zip(figs, populations):
#         ax = fig.add_subplot(111)
#         ax.axis('off')
#         ax.imshow(population[0])
#         fig.tight_layout()

#     return figs
