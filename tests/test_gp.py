import numpy as np
import pymc4 as pm
import pytest

OK = True

BATCH_AND_FEATURE_SHAPES = [(1,), (2,), (2, 2,)]
SAMPLE_SHAPE = [(1,), (5,)]

_check_mean = {
    "Zero": {},
    "Constant": {"coef": 5.0},
}

_check_cov = {
    "ExpQuad": {"amplitude": 1.0, "length_scale": 1.0},
}

_check_gp_model = {"LatentGP": {}}


@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_batch_shape(request):
    return request.param


@pytest.fixture(scope="module", params=SAMPLE_SHAPE, ids=str)
def get_sample_shape(request):
    return request.param


@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_feature_shape(request):
    return request.param


@pytest.fixture(scope="module")
def get_data(get_batch_shape, get_sample_shape, get_feature_shape):
    X = np.random.randn(*(get_batch_shape + get_sample_shape + get_feature_shape)).astype(
        np.float32
    )
    return get_batch_shape, get_sample_shape, get_feature_shape, X


@pytest.fixture(scope="module", params=list(_check_mean), ids=str)
def get_mean_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(_check_cov), ids=str)
def get_cov_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(_check_gp_model), ids=str)
def get_gp_model(request):
    return request.param


@pytest.fixture(scope="module", ids=str)
def get_prior_model():
    @pm.model(name="prior_model")
    def prior_model(gp, X):
        f = yield gp.prior("f", X)
        return f

    return prior_model


@pytest.fixture(scope="module", ids=str)
def get_cond_model():
    @pm.model(name="cond_model")
    def cond_model(gp, X, X_new):
        f = yield gp.prior("f", X)
        fcond = yield gp.conditional("fcond", X_new, given={"X": X, "f": f})
        return fcond

    return cond_model


def get_func(test_dict, test_func, feature_shape, mod):
    kwargs = test_dict[test_func]
    func_class = getattr(mod, test_func)
    func = func_class(feature_ndims=len(feature_shape), **kwargs)
    return func


def test_mean_funcs(tf_seed, get_data, get_mean_func):
    batch_shape, samples_shape, feature_shape, X = get_data
    mean_func = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    mean = mean_func(X)
    assert mean is not None
    assert mean.shape == batch_shape + samples_shape


def test_cov_funcs(tf_seed, get_data, get_cov_func):
    batch_shape, samples_shape, feature_shape, X = get_data
    cov_func = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    cov = cov_func(X, X)
    assert cov is not None
    assert cov.shape.as_list() == list(batch_shape) + list(samples_shape + samples_shape)
    # assert tf.reduce_all(tf.linalg.eigvals(cov) > 0.)
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_gp_models_prior(tf_seed, get_data, get_mean_func, get_cov_func, get_gp_model):
    batch_shape, samples_shape, feature_shape, X = get_data
    mean_func = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    cov_func = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    gp_class = getattr(pm.gp.gp, get_gp_model)
    gp_model = gp_class(mean_fn=mean_func, cov_fn=cov_func)
    try:
        prior_dist = gp_model.prior("prior", X)
        # trace = pm.sample(get_prior_model(gp_model, X), num_samples=3, num_chains=1)
    except NotImplementedError:
        pytest.skip("Skipping: prior not implemented")

    assert prior_dist is not None
    # assert trace.posterior['prior_model/f'] is not None
    if samples_shape == (1,):
        assert prior_dist.sample(1).shape == (1,) + batch_shape
        # assert trace.posterior['prior_model/f'].shape == (1, 3, ) + batch_shape
    else:
        assert prior_dist.sample(1).shape == (1,) + batch_shape + samples_shape
        # assert trace.posterior['prior_model/f'].shape == (1, 3, ) + batch_shape + samples_shape
