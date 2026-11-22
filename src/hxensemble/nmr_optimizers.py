"""NMR reweight_log_lambda and reweight_log_softmax functions."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats

def RMSE(actual, predicted):
    
    # Convert inputs to numpy arrays 
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the squared differences
    squared_diff = (actual - predicted) ** 2
    
    # Compute the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Take the square root of the mean squared difference
    return np.sqrt(mean_squared_diff)

def reweight_log_softmax(
    exp,
    sim_data,
    Neff_threshold=0.1,
    chi2_threshold=0.001,
    w0=None,
    max_iter=50000,
    grad_tol=1e-7,
    mu_floor=1e-300,
    verbose=True,
    theta_grid=None,
    lbfgs_maxcor=10,
    explicit_theta_grid=False,
):
    """
    Reweight in lnPF space using prior-weighted log-softmax parameters
    and L-BFGS-B.

    Model:
        mu_i(w)    = sum_j w_j O_ji
        lnPF_i(w) = -log(mu_i(w))

    Loss:
        L(w) = 0.5 * mean_i [ ((lnPF_calc_i - lnPF_exp_i) / sigma_lnPF_i)^2 ]
             + theta * KL(w || w0)

    where:
        KL(w || w0) = sum_j w_j log(w_j / w0_j)

    Optimisation:
        Prior-weighted log-softmax:
            w_j = w0_j * exp(u_j) / sum_k w0_k * exp(u_k)

        Therefore:
            u = 0 gives w = w0

    Important:
        Each theta starts from the same fixed prior w0.
        No theta warm-starting is used.

    Returns:
        labels, stats_df, wopt_array, wopt,
        np.array([final_chi2, final_phis, final_thetas]),
        rew_lnPF, prior_lnPF
    """

    ############################
    # Helper functions
    ############################

    def entropy_phi(w, wref):
        """Entropy-based effective fraction: exp(-KL(w || wref))."""
        eps = 1e-300
        w_safe = np.clip(w, eps, None)
        wref_safe = np.clip(wref, eps, None)
        kl = np.sum(w_safe * (np.log(w_safe) - np.log(wref_safe)))
        return float(np.exp(-kl))

    def kish_phi(w):
        """Kish effective sample-size fraction."""
        n = len(w)
        return float(1.0 / (n * np.sum(w**2)))

    def weights_from_u(u, wref):
        """
        Prior-weighted log-softmax.

        w_j = wref_j * exp(u_j) / sum_k wref_k * exp(u_k)

        Therefore, if u = 0, weights_from_u(u, wref) == wref.
        """
        u = np.asarray(u, dtype=np.float64)

        # Stability shift. Softmax is invariant to additive constants.
        u_shift = u - np.max(u)

        a = wref * np.exp(u_shift)
        return a / np.sum(a)

    def loss_and_grad_u(u, calc, exp_lnPF, exp_sigma_lnPF, theta, wref):
        """
        Return objective and gradient with respect to prior-weighted
        log-softmax variables u.

        calc shape:
            (n_frames, n_obs)
        """

        # Current weights
        w = weights_from_u(u, wref)

        # Weighted open probability per observable
        mu = w @ calc
        mu = np.clip(mu, mu_floor, None)

        # Predicted lnPF
        pred_lnPF = -np.log(mu)

        # Residuals
        diff = pred_lnPF - exp_lnPF
        inv_sigma2 = 1.0 / (exp_sigma_lnPF * exp_sigma_lnPF)

        # 0.5 * mean(((pred-exp)/sigma)^2)
        chi2 = np.mean(diff * diff * inv_sigma2)
        loss_data = 0.5 * chi2

        # d loss / d pred_lnPF
        dL_dpred = diff * inv_sigma2 / len(exp_lnPF)

        # pred = -log(mu), so d pred / d mu = -1 / mu
        coeff = -dL_dpred / mu

        # d loss / d w_j = sum_i coeff_i * calc_ji
        grad_w = calc @ coeff

        # KL regularisation relative to fixed prior w0
        eps = 1e-300
        wref_safe = np.clip(wref, eps, None)
        ratio = np.clip(w / wref_safe, eps, None)
        log_ratio = np.log(ratio)

        kl = np.sum(w * log_ratio)
        loss_reg = theta * kl

        # Euclidean gradient of KL wrt w
        grad_w += theta * (log_ratio + 1.0)

        loss = loss_data + loss_reg

        # Chain rule through prior-weighted log-softmax.
        #
        # Even with prior-weighted softmax:
        #     w_j = wref_j exp(u_j) / Z
        #
        # the Jacobian wrt u has the same softmax form:
        #     dw_k / du_j = w_k (delta_jk - w_j)
        #
        # therefore:
        #     dL/du_j = w_j * (dL/dw_j - <dL/dw>_w)
        grad_u = w * (grad_w - np.sum(w * grad_w))

        return float(loss), grad_u.astype(np.float64)

    def optimise_for_theta(calc, exp_lnPF, exp_sigma_lnPF, theta, wref):
        """
        Optimise one theta from the fixed prior w0.

        Prior-weighted log-softmax means:
            u0 = 0 gives w = w0

        This is reset for every theta.
        """

        u0 = np.zeros(len(wref), dtype=np.float64)

        options = {
            "maxiter": max_iter,
            "maxfun": max_iter*2,
            "gtol": grad_tol,
            "ftol": 1e-12,
            "maxcor": lbfgs_maxcor,
            "maxls": 20,
            "disp": False,
        }

        result = minimize(
            loss_and_grad_u,
            u0,
            args=(calc, exp_lnPF, exp_sigma_lnPF, theta, wref),
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        w = weights_from_u(result.x, wref).astype(np.float64, copy=True)

        mu = w @ calc
        mu = np.clip(mu, mu_floor, None)
        pred_lnPF = -np.log(mu)

        diff = pred_lnPF - exp_lnPF
        chi2 = np.mean((diff / exp_sigma_lnPF) ** 2)

        phi_entropy = entropy_phi(w, wref)
        phi_kish = kish_phi(w)

        return w, chi2, phi_entropy, phi_kish, pred_lnPF, result

    ############################
    # Prepare simulated data
    ############################

    sim_res = np.asarray(sim_data[0])
    sim_calc = np.asarray(sim_data[1], dtype=np.float64, order="C")

    # Expected shape: (n_frames, n_residues)
    if sim_calc.ndim != 2:
        raise ValueError(
            "sim_data[1] must be a 2D array with shape (n_frames, n_residues)."
        )

    # Residues with no sampled open state cannot produce finite lnPF
    mu_calc = np.mean(sim_calc, axis=0)
    valid = np.where(mu_calc > 0)[0].astype(int)
    invalid = np.where(mu_calc == 0)[0].astype(int)

    invalid_residues = sim_res[invalid]

    sim_res_valid = sim_res[valid]
    sim_calc_valid = sim_calc[:, valid]

    # Common residues between experiment and simulation
    exp_res_array = np.asarray(exp["Model_residue"])
    common_residues = np.intersect1d(sim_res_valid, exp_res_array)

    if len(common_residues) == 0:
        raise ValueError("No common residues between experimental and simulation data.")

    idxscalc = np.array(
        [list(sim_res_valid).index(res) for res in common_residues],
        dtype=int,
    )
    idxsexp = np.array(
        [list(exp_res_array).index(res) for res in common_residues],
        dtype=int,
    )

    labels = np.asarray(exp["Model_residue"]).astype(float)

    # Experimental lnPF values and propagated lnPF errors
    exp_PF = np.asarray(exp["PF"], dtype=np.float64)
    exp_PF_error = np.asarray(exp["PF_error"], dtype=np.float64)

    exp_lnPF = np.log(exp_PF)
    exp_sigma_lnPF = exp_PF_error / exp_PF

    # Subset to matched residues
    calc = np.asarray(sim_calc_valid[:, idxscalc], dtype=np.float64, order="C")
    exp_lnPF = exp_lnPF[idxsexp]
    exp_sigma_lnPF = exp_sigma_lnPF[idxsexp]
    labels = labels[idxsexp]

    # Avoid zero/negative errors
    exp_sigma_lnPF = np.clip(exp_sigma_lnPF, 1e-12, None)

    n_frames = calc.shape[0]

    ############################
    # Prior weights
    ############################

    eps_w = 1e-300

    if w0 is None:
        w0 = np.ones(n_frames, dtype=np.float64) / n_frames
    else:
        w0 = np.asarray(w0, dtype=np.float64)
        if w0.shape[0] != n_frames:
            raise ValueError(
                f"w0 has length {w0.shape[0]}, but calc has {n_frames} frames."
            )
        w0 = np.clip(w0, eps_w, None)
        w0 /= np.sum(w0)

    # warning about residues not sampling open state but present in experimental dataset
    final_invalid_residues = []
    exp_res_list = list(exp["Model_residue"])
    for res in invalid_residues:
        if res in exp_res_list:
            final_invalid_residues.append(res)

    if final_invalid_residues:
        print(
            "Warning: The following residues with experimental PFs have not sampled "
            "the open state: ",
            final_invalid_residues,
        )

    ############################
    # Theta scan
    ############################

    if theta_grid is None:
        theta_grid = list(np.geomspace(1e5, 1e0, 30))
    else:
        theta_grid = list(theta_grid)

    wopt_array = []
    final_thetas = []
    final_phis = []
    final_chi2 = []
    final_kish_phis = []
    final_success = []
    final_messages = []

    def run_theta_list(theta_list):
        nonlocal wopt_array, final_thetas, final_phis, final_chi2
        nonlocal final_kish_phis, final_success, final_messages

        for theta in theta_list:
            # This starts from the fixed prior w0 every time because
            # optimise_for_theta() sets u0 = 0, which gives w = w0.
            w_theta, chi2_theta, phi_theta, kish_theta, pred_lnPF_theta, result = optimise_for_theta(
                calc=calc,
                exp_lnPF=exp_lnPF,
                exp_sigma_lnPF=exp_sigma_lnPF,
                theta=theta,
                wref=w0,
            )

            if verbose:
                print(
                    f"theta={theta:.3e}  chi2_lnPF={chi2_theta:.4f}  "
                    f"phi_entropy={phi_theta:.4f}  phi_kish={kish_theta:.4f}  "
                    f"nit={result.nit}  success={result.success}"
                )

            if chi2_theta > 0.0 and np.all(np.isfinite(w_theta)):
                wopt_array.append(w_theta.astype(np.float64, copy=True))
                final_thetas.append(float(theta))
                final_phis.append(float(phi_theta))
                final_chi2.append(float(chi2_theta))
                final_kish_phis.append(float(kish_theta))
                final_success.append(bool(result.success))
                final_messages.append(str(result.message))

             # stopping logic 
            if not explicit_theta_grid and (phi_theta < Neff_threshold or chi2_theta < chi2_threshold):
                break

    run_theta_list(theta_grid)

    # Fine scan 1
    if not explicit_theta_grid and len(final_phis) > 0 and np.min(final_phis) > Neff_threshold and np.min(final_chi2) > chi2_threshold:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-2, 15)[1:]
        run_theta_list(fine_thetas)

    # Fine scan 2
    if not explicit_theta_grid and len(final_phis) > 0 and np.min(final_phis) > Neff_threshold and np.min(final_chi2) > chi2_threshold:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-6, 15)[1:]
        run_theta_list(fine_thetas)

    if len(wopt_array) == 0:
        raise RuntimeError(
            "No valid reweighting solution found. Try increasing theta range, "
            "lowering Neff_threshold, or checking whether calc contains zeros "
            "for matched experimental residues."
        )

    ############################
    # Select best theta
    ############################

    best_index = -1

    for i, phi in enumerate(final_phis):
        curr_weighted_avg = np.average(calc, axis=0, weights=wopt_array[i])

        if np.all(curr_weighted_avg > 0.0) and phi >= Neff_threshold:
            best_index = i

    if best_index < 0:
        raise RuntimeError("No theta satisfied the Neff/open-state selection criteria.")

    wopt = wopt_array[best_index].astype(np.float64, copy=True)

    # Default adaptive scan retains truncation behaviour.
    # Explicit grids retain every fitted theta (including those beyond phi threshold).
    if not explicit_theta_grid:
        wopt_array = wopt_array[: best_index + 1]
        final_thetas = final_thetas[: best_index + 1]
        final_phis = final_phis[: best_index + 1]
        final_chi2 = final_chi2[: best_index + 1]
        final_kish_phis = final_kish_phis[: best_index + 1]
        final_success = final_success[: best_index + 1]
        final_messages = final_messages[: best_index + 1]

    ############################
    # Statistics
    ############################

    prior_mu = np.average(calc, axis=0, weights=w0)
    prior_mu = np.clip(prior_mu, mu_floor, None)
    prior_lnPF = -np.log(prior_mu)

    rew_mu = np.average(calc, axis=0, weights=wopt)
    rew_mu = np.clip(rew_mu, mu_floor, None)
    rew_lnPF = -np.log(rew_mu)

    # Compare in log10 PF space
    x = exp_lnPF / np.log(10.0)

    y_prior = prior_lnPF / np.log(10.0)
    rpearson_prior = scipy.stats.pearsonr(x, y_prior)[0]
    rmse_prior = RMSE(x, y_prior)

    y_post = rew_lnPF / np.log(10.0)
    rpearson_post = scipy.stats.pearsonr(x, y_post)[0]
    rmse_post = RMSE(x, y_post)

    stats_df = pd.DataFrame()
    stats_df["rpearson prior"] = [rpearson_prior]
    stats_df["rmse prior"] = [rmse_prior]
    stats_df["rpearson post"] = [rpearson_post]
    stats_df["rmse post"] = [rmse_post]
    stats_df["final neff"] = [final_phis[best_index]]
    stats_df["final kish neff"] = [final_kish_phis[best_index]]
    stats_df["final theta"] = [final_thetas[best_index]]
    stats_df["final chi2 lnPF"] = [final_chi2[best_index]]
    stats_df["final optimizer success"] = [final_success[best_index]]
    stats_df["final optimizer message"] = [final_messages[best_index]]
    stats_df["Exp residues no open state"] = [final_invalid_residues]

    return (
        labels,
        stats_df,
        wopt_array,
        wopt,
        np.vstack([
        np.asarray(final_chi2, dtype=np.float64),
        np.asarray(final_phis, dtype=np.float64),
        np.asarray(final_thetas, dtype=np.float64),
        ]),
        rew_lnPF,
        prior_lnPF,
    )

def reweight_log_lambda(
    exp,
    sim_data,
    Neff_threshold=0.1,
    chi2_threshold=0.001,
    w0=None,
    max_iter=50000,
    grad_tol=1e-7,
    mu_floor=1e-300,
    verbose=True,
    theta_grid=None,
    lbfgs_maxcor=10,
    explicit_theta_grid=False,
):
    """
    Reweight in lnPF space using a Gibbs (exponential-family) parametrisation
    of the weights, optimised with L-BFGS-B in the space of one multiplier
    per experimental observable.

    Model:
        mu_i(w)   = sum_j w_j O_ji
        lnPF_i(w) = -log(mu_i(w))

    Loss:
        L(w) = 0.5 * mean_i [ ((lnPF_calc_i - lnPF_exp_i) / sigma_lnPF_i)^2 ]
             + theta * KL(w || w0)

    where:
        KL(w || w0) = sum_j w_j log(w_j / w0_j)

    Parametrisation (the key difference from reweight_log_softmax):
        w_j(lam) = w0_j * exp(-sum_i lam_i O_ji) / Z(lam)

        lam has length n_obs (number of matched experimental observables),
        NOT n_frames. The first-order stationarity condition of the original
        weight-space objective shows that any interior stationary solution must
        have this Gibbs-family form, with lam_i corresponding to effective
        multipliers associated with the observables. Because lnPF is nonlinear
        in w, this is not the conventional linear-observable BME dual problem,
        and global equivalence of the reduced and full weight-space optimisation
        is not formally guaranteed. In practice, the reduced parametrisation
        closely reproduces the solutions obtained by direct weight-space
        optimisation while reducing the optimiser dimensionality from n_frames
        to n_obs. This makes it suitable for very large ensembles (millions of
        frames).

        lam = 0 gives w = w0.

    Starting point:
        Like reweight_log_softmax, every theta starts from the fixed prior w0
        (lam = 0). No warm-starting is used, so a single-theta call returns
        exactly the same weights as that theta within a full theta scan.

    Returns:
        labels, stats_df, wopt_array, wopt,
        np.array([final_chi2, final_phis, final_thetas]),
        rew_lnPF, prior_lnPF
    """

    ############################
    # Helper functions
    ############################

    def entropy_phi(w, wref):
        """Entropy-based effective fraction: exp(-KL(w || wref))."""
        eps = 1e-300
        w_safe = np.clip(w, eps, None)
        wref_safe = np.clip(wref, eps, None)
        kl = np.sum(w_safe * (np.log(w_safe) - np.log(wref_safe)))
        return float(np.exp(-kl))

    def kish_phi(w):
        """Kish effective sample-size fraction."""
        n = len(w)
        return float(1.0 / (n * np.sum(w**2)))

    def weights_from_lambda(lam, calc, wref):
        """
        Gibbs / exponential-family weights.

        w_j = wref_j * exp(-sum_i lam_i O_ji) / sum_k wref_k * exp(-sum_i lam_i O_ki)

        Therefore lam = 0 gives w = wref.
        """
        lam = np.asarray(lam, dtype=np.float64)

        # log-domain score per frame: -O . lam
        t = -(calc @ lam)

        # Stability shift. Normalisation is invariant to an additive constant.
        t -= np.max(t)

        a = wref * np.exp(t)
        return a / np.sum(a)

    def loss_and_grad_lambda(lam, calc, exp_lnPF, exp_sigma_lnPF, theta, wref):
        """
        Return objective and gradient with respect to the observable-space
        multipliers lam.

        calc shape:
            (n_frames, n_obs)
        lam shape:
            (n_obs,)
        """

        # Current weights (Gibbs form)
        w = weights_from_lambda(lam, calc, wref)

        # Weighted open probability per observable
        mu = w @ calc
        mu = np.clip(mu, mu_floor, None)

        # Predicted lnPF
        pred_lnPF = -np.log(mu)

        # Residuals
        diff = pred_lnPF - exp_lnPF
        inv_sigma2 = 1.0 / (exp_sigma_lnPF * exp_sigma_lnPF)

        # 0.5 * mean(((pred-exp)/sigma)^2)
        chi2 = np.mean(diff * diff * inv_sigma2)
        loss_data = 0.5 * chi2

        # d loss / d pred_lnPF
        dL_dpred = diff * inv_sigma2 / len(exp_lnPF)

        # pred = -log(mu), so d pred / d mu = -1 / mu
        coeff = -dL_dpred / mu

        # d loss / d w_j = sum_i coeff_i * O_ji  (Euclidean gradient wrt w)
        grad_w = calc @ coeff

        # KL regularisation relative to fixed prior w0
        eps = 1e-300
        wref_safe = np.clip(wref, eps, None)
        ratio = np.clip(w / wref_safe, eps, None)
        log_ratio = np.log(ratio)

        kl = np.sum(w * log_ratio)
        loss_reg = theta * kl

        # Euclidean gradient of KL wrt w
        grad_w = grad_w + theta * (log_ratio + 1.0)

        loss = loss_data + loss_reg

        # Chain rule through the Gibbs map w_j(lam).
        #
        #   dw_j / dlam_i = -w_j (O_ji - mu_i)
        #
        # therefore, with p_j = w_j * grad_w_j:
        #
        #   dL/dlam_i = sum_j grad_w_j * dw_j/dlam_i
        #             = -(O^T p)_i + mu_i * sum_j p_j
        #
        # (the additive constant from the KL "+1" cancels here, as it must).
        p = w * grad_w
        grad_lambda = -(calc.T @ p) + mu * np.sum(p)

        return float(loss), grad_lambda.astype(np.float64)

    def optimise_for_theta(calc, exp_lnPF, exp_sigma_lnPF, theta, wref, lam0):
        """
        Optimise one theta in observable space, warm-started from lam0.
        """

        n_obs = calc.shape[1]
        # Force a fresh copy: L-BFGS-B may mutate x0 in place, and the same
        # lam0 array is reused across thetas. Copying guarantees every theta
        # genuinely starts from the supplied point (the prior).
        lam0 = np.array(lam0, dtype=np.float64, copy=True)
        if lam0.shape[0] != n_obs:
            lam0 = np.zeros(n_obs, dtype=np.float64)

        options = {
            "maxiter": max_iter,
            "maxfun": max_iter*2,
            "gtol": grad_tol,
            "ftol": 1e-12,
            "maxcor": lbfgs_maxcor,
            "maxls": 20,
            "disp": False,
        }

        result = minimize(
            loss_and_grad_lambda,
            lam0,
            args=(calc, exp_lnPF, exp_sigma_lnPF, theta, wref),
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        w = weights_from_lambda(result.x, calc, wref).astype(np.float64, copy=True)

        mu = w @ calc
        mu = np.clip(mu, mu_floor, None)
        pred_lnPF = -np.log(mu)

        diff = pred_lnPF - exp_lnPF
        chi2 = np.mean((diff / exp_sigma_lnPF) ** 2)

        phi_entropy = entropy_phi(w, wref)
        phi_kish = kish_phi(w)

        return w, chi2, phi_entropy, phi_kish, pred_lnPF, result

    ############################
    # Prepare simulated data
    ############################

    sim_res = np.asarray(sim_data[0])
    sim_calc = np.asarray(sim_data[1], dtype=np.float64, order="C")

    # Expected shape: (n_frames, n_residues)
    if sim_calc.ndim != 2:
        raise ValueError(
            "sim_data[1] must be a 2D array with shape (n_frames, n_residues)."
        )

    # Residues with no sampled open state cannot produce finite lnPF
    mu_calc = np.mean(sim_calc, axis=0)
    valid = np.where(mu_calc > 0)[0].astype(int)
    invalid = np.where(mu_calc == 0)[0].astype(int)

    invalid_residues = sim_res[invalid]

    sim_res_valid = sim_res[valid]
    sim_calc_valid = sim_calc[:, valid]

    # Common residues between experiment and simulation
    exp_res_array = np.asarray(exp["Model_residue"])
    common_residues = np.intersect1d(sim_res_valid, exp_res_array)

    if len(common_residues) == 0:
        raise ValueError("No common residues between experimental and simulation data.")

    idxscalc = np.array(
        [list(sim_res_valid).index(res) for res in common_residues],
        dtype=int,
    )
    idxsexp = np.array(
        [list(exp_res_array).index(res) for res in common_residues],
        dtype=int,
    )

    labels = np.asarray(exp["Model_residue"]).astype(float)

    # Experimental lnPF values and propagated lnPF errors
    exp_PF = np.asarray(exp["PF"], dtype=np.float64)
    exp_PF_error = np.asarray(exp["PF_error"], dtype=np.float64)

    exp_lnPF = np.log(exp_PF)
    exp_sigma_lnPF = exp_PF_error / exp_PF

    # Subset to matched residues
    calc = np.asarray(sim_calc_valid[:, idxscalc], dtype=np.float64, order="C")
    exp_lnPF = exp_lnPF[idxsexp]
    exp_sigma_lnPF = exp_sigma_lnPF[idxsexp]
    labels = labels[idxsexp]

    # Avoid zero/negative errors
    exp_sigma_lnPF = np.clip(exp_sigma_lnPF, 1e-12, None)

    n_frames = calc.shape[0]
    n_obs = calc.shape[1]

    ############################
    # Prior weights
    ############################

    eps_w = 1e-300

    if w0 is None:
        w0 = np.ones(n_frames, dtype=np.float64) / n_frames
    else:
        w0 = np.asarray(w0, dtype=np.float64)
        if w0.shape[0] != n_frames:
            raise ValueError(
                f"w0 has length {w0.shape[0]}, but calc has {n_frames} frames."
            )
        w0 = np.clip(w0, eps_w, None)
        w0 /= np.sum(w0)

    # warning about residues not sampling open state but present in experimental dataset
    final_invalid_residues = []
    exp_res_list = list(exp["Model_residue"])
    for res in invalid_residues:
        if res in exp_res_list:
            final_invalid_residues.append(res)

    if final_invalid_residues:
        print(
            "Warning: The following residues with experimental PFs have not sampled "
            "the open state: ",
            final_invalid_residues,
        )

    ############################
    # Theta scan
    ############################

    if theta_grid is None:
        theta_grid = list(np.geomspace(1e5, 1e0, 30))
    else:
        theta_grid = list(theta_grid)

    wopt_array = []
    final_thetas = []
    final_phis = []
    final_chi2 = []
    final_kish_phis = []
    final_success = []
    final_messages = []

    # Each theta starts from the fixed prior w0 (lam = 0). No warm-starting,
    # so a single-theta call returns exactly the same weights as that theta
    # within a full theta scan.
    lam0_fixed = np.zeros(n_obs, dtype=np.float64)  # lam = 0  <=>  w = w0

    def run_theta_list(theta_list):
        nonlocal wopt_array, final_thetas, final_phis, final_chi2
        nonlocal final_kish_phis, final_success, final_messages

        for theta in theta_list:
            w_theta, chi2_theta, phi_theta, kish_theta, pred_lnPF_theta, result = optimise_for_theta(
                calc=calc,
                exp_lnPF=exp_lnPF,
                exp_sigma_lnPF=exp_sigma_lnPF,
                theta=theta,
                wref=w0,
                lam0=lam0_fixed,
            )

            if verbose:
                print(
                    f"theta={theta:.3e}  chi2_lnPF={chi2_theta:.4f}  "
                    f"phi_entropy={phi_theta:.4f}  phi_kish={kish_theta:.4f}  "
                    f"nit={result.nit}  success={result.success}"
                )

            if chi2_theta > 0.0 and np.all(np.isfinite(w_theta)):
                wopt_array.append(w_theta.astype(np.float64, copy=True))
                final_thetas.append(float(theta))
                final_phis.append(float(phi_theta))
                final_chi2.append(float(chi2_theta))
                final_kish_phis.append(float(kish_theta))
                final_success.append(bool(result.success))
                final_messages.append(str(result.message))

            # Same stopping logic as reweight_log_softmax
            if not explicit_theta_grid and (phi_theta < Neff_threshold or chi2_theta < chi2_threshold):
                break

    run_theta_list(theta_grid)

    # Fine scan 1
    if not explicit_theta_grid and len(final_phis) > 0 and np.min(final_phis) > Neff_threshold and np.min(final_chi2) > chi2_threshold:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-2, 15)[1:]
        run_theta_list(fine_thetas)

    # Fine scan 2
    if not explicit_theta_grid and len(final_phis) > 0 and np.min(final_phis) > Neff_threshold and np.min(final_chi2) > chi2_threshold:
        fine_thetas = np.geomspace(final_thetas[-1], 1e-6, 15)[1:]
        run_theta_list(fine_thetas)

    if len(wopt_array) == 0:
        raise RuntimeError(
            "No valid reweighting solution found. Try increasing theta range, "
            "lowering Neff_threshold, or checking whether calc contains zeros "
            "for matched experimental residues."
        )

    ############################
    # Select best theta
    ############################

    best_index = -1

    for i, phi in enumerate(final_phis):
        curr_weighted_avg = np.average(calc, axis=0, weights=wopt_array[i])

        if np.all(curr_weighted_avg > 0.0) and phi >= Neff_threshold:
            best_index = i

    if best_index < 0:
        raise RuntimeError("No theta satisfied the Neff/open-state selection criteria.")

    wopt = wopt_array[best_index].astype(np.float64, copy=True)

    # Default adaptive scan retains original truncation behaviour.
    # Explicit grids retain every fitted theta (including those beyond phi threshold).
    if not explicit_theta_grid:
        wopt_array = wopt_array[: best_index + 1]
        final_thetas = final_thetas[: best_index + 1]
        final_phis = final_phis[: best_index + 1]
        final_chi2 = final_chi2[: best_index + 1]
        final_kish_phis = final_kish_phis[: best_index + 1]
        final_success = final_success[: best_index + 1]
        final_messages = final_messages[: best_index + 1]

    ############################
    # Statistics
    ############################

    prior_mu = np.average(calc, axis=0, weights=w0)
    prior_mu = np.clip(prior_mu, mu_floor, None)
    prior_lnPF = -np.log(prior_mu)

    rew_mu = np.average(calc, axis=0, weights=wopt)
    rew_mu = np.clip(rew_mu, mu_floor, None)
    rew_lnPF = -np.log(rew_mu)

    # Compare in log10 PF space
    x = exp_lnPF / np.log(10.0)

    y_prior = prior_lnPF / np.log(10.0)
    rpearson_prior = scipy.stats.pearsonr(x, y_prior)[0]
    rmse_prior = RMSE(x, y_prior)

    y_post = rew_lnPF / np.log(10.0)
    rpearson_post = scipy.stats.pearsonr(x, y_post)[0]
    rmse_post = RMSE(x, y_post)

    stats_df = pd.DataFrame()
    stats_df["rpearson prior"] = [rpearson_prior]
    stats_df["rmse prior"] = [rmse_prior]
    stats_df["rpearson post"] = [rpearson_post]
    stats_df["rmse post"] = [rmse_post]
    stats_df["final neff"] = [final_phis[best_index]]
    stats_df["final kish neff"] = [final_kish_phis[best_index]]
    stats_df["final theta"] = [final_thetas[best_index]]
    stats_df["final chi2 lnPF"] = [final_chi2[best_index]]
    stats_df["final optimizer success"] = [final_success[best_index]]
    stats_df["final optimizer message"] = [final_messages[best_index]]
    stats_df["Exp residues no open state"] = [final_invalid_residues]

    return (
        labels,
        stats_df,
        wopt_array,
        wopt,
        np.vstack([
        np.asarray(final_chi2, dtype=np.float64),
        np.asarray(final_phis, dtype=np.float64),
        np.asarray(final_thetas, dtype=np.float64),
        ]),
        rew_lnPF,
        prior_lnPF,
    )
