#include<RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;
using namespace std;

//[[Rcpp::depends(RcppArmadillo)]]
uvec uvunion(uvec& first, uvec& second) {
	uvec res(first.n_elem + second.n_elem);
	uvec::iterator fb = first.begin(), sb = second.begin(), se = second.end(), fe = first.end(), rb = res.begin();
	int j = 0;
	if (*fb == *sb) {
		*rb++ = *fb++;
		++sb;
		++j;
	}
	while (true) {
		if (sb == se) {
			if (*fb == *rb) {
				++fb;
			}
			while (fb < fe) {
				*rb++ = *fb++;
				++j;
			}
			break;
		}
		if (fb == fe) {
			if (*sb == *rb) {
				++sb;
			}
			while (sb < se) {
				*rb++ = *sb++;
				++j;
			}
			break;
		}
		if (*fb > *sb) {
			*rb++ = *sb++;
			++j;
		}
		else if (*fb < *sb) {
			*rb++ = *fb++;
			++j;
		}
		else {
			if (*fb == *rb) {
				++fb;
				++sb;
			}
			else {
				*rb++ = *fb++;
				++j;
				++sb;
			}

		}
	}
	return res.head(j);
}

uvec setdiff(uvec& bi, uvec& sm) {
	uvec res(bi.n_elem - sm.n_elem);
	uvec::iterator sb = sm.begin(), bb = bi.begin(), rb = res.begin(), se = sm.end(), be = bi.end();
	while (bb < be) {
		if (sb == se) {
			*rb++ = *bb++;
		}
		else if (*bb == *sb) {
			++bb;
			++sb;
		}
		else {
			*rb++ = *bb++;
		}
	}
	return res;
}

inline mat HessSVM(mat& ct, vec& yapp, mat& wvec) {
	vec tmp;
	mat hb = zeros(yapp.n_elem, yapp.n_elem);
	uvec ind1 = find(wvec.col(0) > 0), ind2 = find(wvec.col(1) > 0);
	if (ind1.n_elem > 0) {
		tmp = ct.col(0);
		hb(ind1, ind1) = diagmat(tmp(ind1));
	}
	if (ind2.n_elem > 0) {
		tmp = ct.col(1);
		hb(ind2, ind2) += diagmat(tmp(ind2));
	}
	return hb;
}


inline vec gradSVM(mat& wvec, vec& yapp) {
	return (1 + wvec.col(1) - wvec.col(0)) % yapp;
}

inline mat HessLogit(vec& yrho) {
	return diagmat(-1 / (yrho % (1 + yrho)));
}

inline vec gradLogit(vec& yrho, vec& yapp) {
	return yapp%log(-(1 + yrho) / yrho);
}

inline void setzero(vec& v) {
	uvec tmp = find(v < 0);
	if (tmp.n_elem > 0) {
		v(tmp) = zeros(tmp.n_elem);
	}
}

inline void setone(vec& v) {
	uvec tmp = find(v < -1);
	if (tmp.n_elem > 0) {
		v(tmp) = -1 * ones(tmp.n_elem);
	}
}

inline void set9901(vec& v) {
	uvec tmp = find(v < -.9999999);
	if (tmp.n_elem > 0) {
		v(tmp) = -.9999999*ones(tmp.n_elem);
	}
	tmp = find(v > -.00000001);
	if (tmp.n_elem > 0) {
		v(tmp) = -.00000001*ones(tmp.n_elem);
	}
}



inline void set1(vec& v) {
	uvec tmp = find(v > 1);
	if (tmp.n_elem > 0) {
		v(tmp) = ones(tmp.n_elem);
	}
}

inline vec pmax(vec& a1, vec& a2) {
	vec a(a1.n_elem);
	vec::iterator a1b = a1.begin(), a2b = a2.begin(), a1e = a1.end(), ab = a.begin();
	while (a1b < a1e) {
		*ab++ = max(*a1b++, *a2b++);
	}
	return a;
}

inline double losssvm(vec& yrho) {
	return sum(yrho);
}

inline double losslogit(vec& yrho) {
	return sum((1 + yrho) % log(1 + yrho) - yrho%log(-yrho));
}

inline vec regfunc(vec& x, double C) {
	return C*abs(x);
}

inline vec regfuncdual(vec& x, double C) {
	vec regd = zeros(x.n_elem);
	uvec goinf = find(x > C);
	if (goinf.n_elem > 0) {
		regd(goinf) = datum::inf*ones(goinf.n_elem);
	}
	return regd;
}

inline vec prox(vec& x, double C, vec& eta) {
	vec prx = x - C*eta;
	setzero(prx);
	return prx;
}

inline vec dprox(vec& x, double C, vec& eta) {
	return conv_to<vec>::from(x > C*eta);
}

inline mat norm3kjv(cube& K, mat& u, uvec& acc) {
	mat v(u.n_rows, acc.n_elem);
	uvec::iterator ab = acc.begin();
	for (unsigned i = 0; i < acc.n_elem; ++i) {
		v.col(i) = K.slice(*ab)*u.col(*ab);
		++ab;
	}
	return v;
}

inline vec norm3kjw(cube& K, mat& u, mat& v, uvec& acc) {
	vec wj(acc.n_elem);
	vec::iterator wb = wj.begin();
	uvec::iterator ab = acc.begin();
	for (unsigned i = 0; i < acc.n_elem; ++i) {
		*wb++ = sqrt(dot(v.col(i), u.col(*ab++)));
	}
	return wj;
}

inline mat norm2kjv(cube& K, mat& u) {
	mat v(u.n_rows, u.n_cols);
	for (unsigned i = 0; i < u.n_cols; ++i) {
		v.col(i) = K.slice(i)*u.col(i);
	}
	return v;
}

inline vec norm2kjw(cube& K, mat& u, mat& v) {
	vec wj(u.n_cols);
	vec::iterator wb = wj.begin();
	for (unsigned i = 0; i < u.n_cols; ++i) {
		*wb++ = sqrt(dot(v.col(i), u.col(i)));
	}
	return wj;
}

inline void HessAugMexbias(cube& K, vec& wj, uvec& activeset, mat& v, double C, vec& cgamma, double cgammab, mat& Hessian, vec& alpha1, vec& alpha2) {
	uvec::iterator ab = activeset.begin(), ae = activeset.end();
	while (ab < ae) {
		Hessian += alpha1(*ab)*K.slice(*ab) + alpha2(*ab)*v.col(*ab)*v.col(*ab).t();
		++ab;
	}
	Hessian += cgammab;
}

inline mat getwvec(mat& clambda, mat& ct, vec& yrho) {
	mat wvec(clambda.n_rows, 2);
	vec tmp = clambda.col(0) - ct.col(0) % (1 + yrho);
	setzero(tmp);
	wvec.col(0) = tmp;
	tmp = clambda.col(1) + ct.col(1) % yrho;
	setzero(tmp);
	wvec.col(1) = tmp;
	return wvec;
}

inline double funcevalsvm(mat& wvec, vec& wj, vec& yapp, vec& rho, vec& yrho, double sumrho, vec& cgamma, double cgammab, double cb, vec& qm, double C, mat& clambda, mat& ct) {
	return losssvm(yrho) - sum(regfunc(qm, C)) - .5*sum(pow(qm, 2) / cgamma) + dot(wj, qm) + .5*cgammab*pow(sumrho, 2) + cb*sumrho + .5*accu((pow(wvec.col(0), 2) - pow(clambda.col(0), 2)) / ct.col(0) + (pow(wvec.col(1), 2) - pow(clambda.col(1), 2)) / ct.col(1));
}

inline double funcevallogit(vec& wj, vec& yapp, vec& rho, vec& yrho, double sumrho, vec& cgamma, double cgammab, double cb, vec& qm, double C) {
	return losslogit(yrho) - sum(regfunc(qm, C)) - .5*sum(pow(qm, 2) / cgamma) + dot(wj, qm) + .5*cgammab*pow(sumrho, 2) + cb*sumrho;
}

inline vec gradientsvm(vec& yapp, vec& yrho, vec& rho, mat& v, vec& cgamma, uvec& activeset, vec& wj, double C, double sumrho, vec& qm, mat& wvec, double cgammab, double cb) {
	vec grad = gradSVM(wvec, yapp);
	uvec::iterator ab = activeset.begin(), ae = activeset.end();
	while (ab < ae) {
		grad += qm(*ab) / wj(*ab)*v.col(*ab);
		++ab;
	}
	grad += cgammab*sumrho + cb;
	return grad;
}

inline vec gradientlogit(vec& yapp, vec& yrho, vec& rho, mat& v, vec& cgamma, uvec& activeset, vec& wj, double C, double sumrho, vec& qm, double cgammab, double cb) {
	vec grad = gradLogit(yrho, yapp);
	uvec::iterator ab = activeset.begin(), ae = activeset.end();
	while (ab < ae) {
		grad += qm(*ab) / wj(*ab)*v.col(*ab);
		++ab;
	}
	grad += cgammab*sumrho + cb;
	return grad;
}

//[[Rcpp::export]]
List SpicySVM(arma::cube K, arma::vec yapp, double C, double tol, double tolInner, int OuterMaxIter, int InnerMaxIter, double calpha) {
	bool b_cb;
	int N = yapp.n_elem, M = K.n_slices;
	double cgammab = 1, cb = 0, ck = numeric_limits<double>::infinity(), cbeta = .5, fval, sumrho, graddotd, old_fval, steplen, mod_dualobj, primalobj, maxgap, work0, sumd, aa2, stc, crhowj;
	vec rho = -.5*yapp, cgamma = 10 * ones(M), dirnorm, dirdotu, yrho, wj, grad, alpha1, alpha2, dk, old_rho, old_wj, old_yrho, dir, wjtmp, dqm, muwj, modrho, rhowj, aa, ay, modyrho, hresid, cw1, ksi, eta, work, d, qm, tmp;
	mat mu = zeros(N, M), ct = 10 * ones(N, 2), clambda = zeros(N, 2), u = mu.each_col() + rho, v, wvec, Hessian, old_u, wvec2, old_v, vmu, vv, modrho1, uu, alpha, tmp_mat;
	uvec allind = conv_to<uvec>::from(linspace(0, N - 1, N)), allind2 = conv_to<uvec>::from(linspace(0, M - 1, M)), supind = allind, activeset, old_activeset, tmp_activeset, actdif, I3, ctI1, ctI2, cI3;
	uvec::iterator ab, ae;
	List res;
	v = norm2kjv(K, u);
	wj = norm2kjw(K, u, v);
	tmp = wj%cgamma;
	qm = prox(tmp, C, cgamma);
	activeset = find(qm > 0);
	for (int l = 0; l < OuterMaxIter; ++l) {
		if (activeset.n_elem == 0) {
			break;
		}
		for (int step = 0; step < InnerMaxIter; ++step) {
			sumrho = sum(rho);
			yrho = yapp%rho;
			wvec = getwvec(clambda, ct, yrho);
			fval = funcevalsvm(wvec, wj, yapp, rho, yrho, sumrho, cgamma, cgammab, cb, qm, C, clambda, ct);
			grad = gradientsvm(yapp, yrho, rho, v, cgamma, activeset, wj, C, sumrho, qm, wvec, cgammab, cb);
			tmp = wj%cgamma;
			dqm = dprox(tmp, C, cgamma);
			alpha2 = (cgamma%dqm%wj - qm) / pow(wj, 3);
			alpha1 = qm / wj;
			Hessian = HessSVM(ct, yapp, wvec);
			HessAugMexbias(K, wj, activeset, v, C, cgamma, cgammab, Hessian, alpha1, alpha2);
			dk = -solve(Hessian + 1e-8*eye(N, N), grad);
			graddotd = dot(grad, dk);
			if (graddotd > 0) {
				dk = -grad;
			}
			old_fval = fval;
			old_rho = rho;
			old_wj = wj;
			old_u = u;
			old_v = v;
			old_activeset = activeset;
			old_yrho = yrho;
			rho = old_rho + dk;
			u = mu.each_col() + rho;
			v = norm2kjv(K, u);
			wj = norm2kjw(K, u, v);
			tmp = wj%cgamma;
			qm = prox(tmp, C, cgamma);
			activeset = find(qm > 0);
			if (activeset.n_elem == 0) {
				break;
			}
			sumrho = sum(rho);
			tmp = yapp%rho;
			wvec2 = getwvec(clambda, ct, tmp);
			fval = funcevalsvm(wvec2, wj, yapp, rho, tmp, sumrho, cgamma, cgammab, cb, qm, C, clambda, ct);
			dir = dk;
			tmp_activeset = uvunion(activeset, old_activeset);
			
			if (tmp_activeset.n_elem > activeset.n_elem) {
				actdif = setdiff(tmp_activeset, activeset);
				tmp_mat = mu.cols(actdif);
				old_u.cols(actdif) = tmp_mat.each_col() + old_rho;
				old_v.cols(actdif) = norm3kjv(K, old_u, actdif);
				tmp_mat = old_v.cols(actdif);
				old_wj(actdif) = norm3kjw(K, old_u, tmp_mat, actdif);
			}
			dirnorm = zeros(M);
			dirdotu = zeros(M);
			dirnorm(tmp_activeset) = trans(v.cols(tmp_activeset) - old_v.cols(tmp_activeset))*dir;
			dirdotu(tmp_activeset) = trans(old_v.cols(tmp_activeset))*dir;
			steplen = 1;
			stc = .1*dot(dir, grad);
			while (fval > old_fval + steplen*stc && steplen > 0) {
				steplen *= .5;
				rho = old_rho + steplen*dir;
				wj = zeros(M);
				wjtmp = pow(old_wj(tmp_activeset), 2) + 2 * steplen*dirdotu(tmp_activeset) + pow(steplen, 2)*dirnorm(tmp_activeset);
				setzero(wjtmp);
				wj(tmp_activeset) = sqrt(wjtmp);
				tmp = wj%cgamma;
				qm = prox(tmp, C, cgamma);
				tmp = yapp%rho;
				wvec2 = getwvec(clambda, ct, tmp);
				fval = funcevalsvm(wvec2, wj, yapp, rho, tmp, sum(rho), cgamma, cgammab, cb, qm, C, clambda, ct);
			}
			if (steplen != 1) {
				activeset = find(qm > 0);
				if (activeset.n_elem == 0) {
					break;
				}
				u = mu.each_col() + rho;
				v.cols(activeset) = norm3kjv(K, u, activeset);
			}
			if (norm(old_rho - rho) / norm(old_rho) <= tolInner) {
				break;
			}
			if (step == InnerMaxIter - 1) {
                Rcout << "Does not converge in inner cycle."<<endl;
			}
		}
		sumrho = sum(rho);
		activeset = find(qm > 0);
		if (activeset.n_elem == 0) {
			break;
		}
		yrho = yapp%rho;
		mu = zeros(N, M);
		tmp_mat = u.cols(activeset);
		mu.cols(activeset) = tmp_mat.each_row() % trans (qm(activeset) / wj(activeset) / cgamma(activeset));
		cb += cgammab*sumrho;
		alpha = mu.each_row() % trans(-cgamma);
		vmu = norm3kjv(K, alpha, activeset);
		muwj = norm3kjw(K, alpha, vmu, activeset);
		modrho = rho - mean(rho);
		modrho1 = repelem(modrho, 1, M);
		vv = norm3kjv(K, modrho1, activeset);
		rhowj = norm3kjw(K, modrho1, vv, activeset);
		crhowj = min(1.0, C / max(rhowj));
		modrho *= crhowj;
		rhowj *= crhowj;
		aa = zeros(N);
		supind = find(clambda.col(1) == 0);
		if (supind.n_elem > 0) {
			ab = activeset.begin();
			ae = activeset.end();
			while (ab < ae) {
				tmp = alpha.col(*ab);
				aa -= K.slice(*ab).cols(supind)*tmp(supind);
				++ab;
			}
		}
		aa += cb;
		ay = aa%yapp;
		setone(ay);
		aa2 = sum(ay + 1);
		modyrho = yapp%modrho;
		mod_dualobj = -losssvm(modyrho) - sum(regfuncdual(rhowj, C));
		primalobj = aa2 + sum(regfunc(muwj, C));
		uu = u;
		cw1 = C / wj(activeset);
		set1(cw1);
		tmp_mat = u.cols(activeset);
		uu.cols(activeset) = tmp_mat.each_row() % trans(cw1);
		hresid = sqrt(trans(sum(pow(uu.each_col() - rho, 2))));
		maxgap = 0;
		ksi = -clambda.col(0) / ct.col(0);
		tmp = -1 - yrho;
		work = abs(pmax(tmp, ksi));
		ctI1 = find(work > cbeta*ck);
		maxgap = max(max(work), maxgap);
		eta = -clambda.col(1) / ct.col(1);
		work = abs(pmax(yrho, eta));
		ctI2 = find(work > cbeta*ck);
		maxgap = max(max(work), maxgap);
		work0 = fabs(sumrho);
		b_cb = work0 > cbeta*ck;
		maxgap = max(work0, maxgap);
		I3 = find(hresid > cbeta*ck);
		maxgap = max(max(abs(hresid)), maxgap);
		if (maxgap <= ck) {
			ck = maxgap;
		}
		ksi = clambda.col(0) - ct.col(0) % (1 + yrho);
		eta = clambda.col(1) + ct.col(1) % yrho;
		setzero(ksi);
		setzero(eta);
		clambda.col(0) = ksi;
		clambda.col(1) = eta;
		ct += calpha;
		if (ctI1.n_elem > 0) {
			ab = ctI1.begin();
			ae = ctI1.end();
			while (ab < ae) {
				ct(*ab, 0) = (ct(*ab, 0) - calpha)*calpha;
				++ab;
			}
		}
		if (ctI2.n_elem > 0) {
			ab = ctI2.begin();
			ae = ctI2.end();
			while (ab < ae) {
				ct(*ab, 1) = (ct(*ab, 1) - calpha)*calpha;
				++ab;
			}
		}
		if (b_cb) {
			cgammab *= calpha;
		}
		else {
			cgammab += calpha;
		}
		if (I3.n_elem > 0) {
			cgamma(I3) *= calpha;
			mu.cols(I3) /= calpha;
		}
		if (I3.n_elem < (unsigned)M) {
			if (I3.n_elem == 0) {
				cI3 = allind2;
			}
			else {
				cI3 = setdiff(allind2, I3);
			}
			cgamma(cI3) += calpha;
			tmp_mat = mu.cols(cI3);
			tmp_mat.each_row() %= trans((cgamma(cI3) - calpha) / cgamma(cI3));
			mu.cols(cI3) = tmp_mat;
		}
		u = mu.each_col() + rho;
		v = norm2kjv(K, u);
		wj = norm2kjw(K, u, v);
		tmp = wj%cgamma;
		qm = prox(tmp, C, cgamma);
		activeset = find(qm > 0);
		if (activeset.n_elem == 0 || fabs((primalobj - mod_dualobj) / primalobj) < tol) {
			break;
		}
		if (l == OuterMaxIter) {
            Rcout << "Does not converge in outer cycle." << endl;
		}
	}
	work = zeros(M);
	if (activeset.n_elem > 0) {
		work(activeset) = qm(activeset) / wj(activeset);
	}
	else {
		cb = -mean(yapp);
	}
	d = work / (1 - work / cgamma);
	sumd = sum(d);
	if (sumd != 0) {
		d /= sumd;
	}
	res["alpha"]= -sumd*rho*d.t();
	res["b"] = -cb;
	res["weight"] = d;
	res["rho"] = -sumd*rho;
	return res;
}

//[[Rcpp::export]]
List SpicyLogit(arma::cube K, arma::vec yapp, double C, double tol, double tolInner, int OuterMaxIter, int InnerMaxIter, double calpha) {
	bool b_cb;
	int N = yapp.n_elem, M = K.n_slices;
	double cgammab = 1, cb = 0, ck = datum::inf, cbeta = .5, fval, sumrho, graddotd, old_fval, steplen, mod_dualobj, primalobj, maxgap, sumd, aa2, stc, crhowj;
	vec rho = -.5*yapp, cgamma = 10 * ones(M), ss(3), dirnorm, dirdotu, yrho, wj, grad, alpha1, alpha2, dk, old_rho, old_wj, old_yrho, dir, wjtmp, dqm, muwj, modrho, rhowj, aa, modyrho, hresid, cw1, d, qm, tmp, yd, work;
	mat mu = zeros(N, M), u = mu.each_col() + rho, v, Hessian, old_u, old_v, vmu, vv, modrho1, uu, alpha, tmp_mat;
	uvec allind2 = conv_to<uvec>::from(linspace(0, M - 1, M)), activeset, old_activeset, tmp_activeset, actdif, I3, cI3, yd0;
	uvec::iterator ab, ae;
	List res;
	ss(0) = 1;
	v = norm2kjv(K, u);
	wj = norm2kjw(K, u, v);
	tmp = wj%cgamma;
	qm = prox(tmp, C, cgamma);
	activeset = find(qm > 0);
	for (int l = 0; l < OuterMaxIter; ++l) {
		if (activeset.n_elem == 0) {
			break;
		}
		for (int step = 0; step < InnerMaxIter; ++step) {
			sumrho = sum(rho);
			yrho = yapp%rho;
			fval = funcevallogit(wj, yapp, rho, yrho, sumrho, cgamma, cgammab, cb, qm, C);
			grad = gradientlogit(yapp, yrho, rho, v, cgamma, activeset, wj, C, sumrho, qm, cgammab, cb);
			tmp = wj%cgamma;
			dqm = dprox(tmp, C, cgamma);
			alpha2 = (cgamma%dqm%wj - qm) / pow(wj, 3);
			alpha1 = qm / wj;
			Hessian = HessLogit(yrho);
			HessAugMexbias(K, wj, activeset, v, C, cgamma, cgammab, Hessian, alpha1, alpha2);
			dk = -solve(Hessian, grad);
			graddotd = dot(grad, dk);
			if (graddotd > 0) {
				dk = -grad;
			}
			old_fval = fval;
			old_rho = rho;
			old_wj = wj;
			old_u = u;
			old_v = v;
			old_activeset = activeset;
			old_yrho = yrho;
			rho = old_rho + dk;
			yrho = rho%yapp;
			if (any(yrho <= -1 || yrho >= 0)) {
				yd = yapp%dk;
				yd0 = find(yd < 0);
				ss(1) = yd0.n_elem > 0 ? (.99*min(-(1 + old_yrho(yd0)) / yd(yd0))) : 1;
				yd0 = find(yd > 0);
				ss(2) = yd0.n_elem > 0 ? (.99*min(-old_yrho(yd0) / yd(yd0))) : 1;
				rho = old_rho + min(ss)*dk;
			}
			u = mu.each_col() + rho;
			v = norm2kjv(K, u);
			wj = norm2kjw(K, u, v);
			tmp = wj%cgamma;
			qm = prox(tmp, C, cgamma);
			activeset = find(qm > 0);
			if (activeset.n_elem == 0) {
				break;
			}
			sumrho = sum(rho);
			tmp = yapp%rho;
			fval = funcevallogit(wj, yapp, rho, tmp, sumrho, cgamma, cgammab, cb, qm, C);
			dir = rho - old_rho;
			tmp_activeset = uvunion(activeset, old_activeset);
			if (tmp_activeset.n_elem > activeset.n_elem) {
				actdif = setdiff(tmp_activeset, activeset);
				tmp_mat = mu.cols(actdif);
				old_u.cols(actdif) = tmp_mat.each_col() + old_rho;
				old_v.cols(actdif) = norm3kjv(K, old_u, actdif);
				tmp_mat = old_v.cols(actdif);
				old_wj(actdif) = norm3kjw(K, old_u, tmp_mat, actdif);
			}
			dirnorm = zeros(M);
			dirdotu = zeros(M);
			dirnorm(tmp_activeset) = trans(v.cols(tmp_activeset) - old_v.cols(tmp_activeset))*dir;
			dirdotu(tmp_activeset) = trans(old_v.cols(tmp_activeset))*dir;
			steplen = 1;
			stc = .1*dot(dir, grad);
			while (fval > old_fval + steplen*stc && steplen > 0) {
				steplen *= .5;
				rho = old_rho + steplen*dir;
				wj = zeros(M);
				wjtmp = pow(old_wj(tmp_activeset), 2) + 2 * steplen*dirdotu(tmp_activeset) + pow(steplen, 2)*dirnorm(tmp_activeset);
				setzero(wjtmp);
				wj(tmp_activeset) = sqrt(wjtmp);
				tmp = wj%cgamma;
				qm = prox(tmp, C, cgamma);
				tmp = yapp%rho;
				fval = funcevallogit(wj, yapp, rho, tmp, sum(rho), cgamma, cgammab, cb, qm, C);
			}
			if (steplen != 1) {
				activeset = find(qm > 0);
				if (activeset.n_elem == 0) {
					break;
				}
				u = mu.each_col() + rho;
				v.cols(activeset) = norm3kjv(K, u, activeset);
			}
			if (norm(old_rho - rho) / norm(old_rho) <= tolInner) {
				break;
			}
			if (step == InnerMaxIter - 1) {
				Rcout << "Does not converge in inner cycle." <<endl;
			}
		}
		sumrho = sum(rho);
		activeset = find(qm > 0);
		if (activeset.n_elem == 0) {
			break;
		}
		yrho = yapp%rho;
		mu = zeros(N, M);
		tmp_mat = u.cols(activeset);
		mu.cols(activeset) = tmp_mat.each_row() % trans(qm(activeset) / wj(activeset) / cgamma(activeset));
		cb += cgammab*sumrho;
		alpha = mu.each_row() % trans(-cgamma);
		vmu = norm3kjv(K, alpha, activeset);
		muwj = norm3kjw(K, alpha, vmu, activeset);
		modrho = rho - mean(rho);
		modrho1 = repelem(modrho, 1, M);
		vv = norm3kjv(K, modrho1, activeset);
		rhowj = norm3kjw(K, modrho1, vv, activeset);
		crhowj = min(1.0, C / max(rhowj));
		modrho *= crhowj;
		rhowj *= crhowj;
		aa = zeros(N);
		ab = activeset.begin();
		ae = activeset.end();
		while (ab < ae) {
			aa -= K.slice(*ab)*alpha.col(*ab);
			++ab;
		}
		aa += cb;
		aa2 = sum(log(1 + exp(aa%yapp)));
		modyrho = yapp%modrho;
		set9901(modyrho);
		mod_dualobj = -losslogit(modyrho) - sum(regfuncdual(rhowj, C));
		primalobj = aa2 + sum(regfunc(muwj, C));
		uu = u;
		cw1 = C / wj(activeset);
		set1(cw1);
		tmp_mat = u.cols(activeset);
		uu.cols(activeset) = tmp_mat.each_row() % trans(cw1);
		hresid = sqrt(trans(sum(pow(uu.each_col() - rho, 2))));
		maxgap = fabs(sumrho);
		b_cb = maxgap > cbeta*ck;
		I3 = find(hresid > cbeta*ck);
		maxgap = max(max(abs(hresid)), maxgap);
		if (maxgap <= ck) {
			ck = maxgap;
		}
		if (b_cb) {
			cgammab *= calpha;
		}
		else {
			cgammab += calpha;
		}
		if (I3.n_elem > 0) {
			cgamma(I3) *= calpha;
			mu.cols(I3) /= calpha;
		}
		if (I3.n_elem < (unsigned)M) {
			if (I3.n_elem == 0) {
				cI3 = allind2;
			}
			else {
				cI3 = setdiff(allind2, I3);
			}
			cgamma(cI3) += calpha;
			tmp_mat = mu.cols(cI3);
			tmp_mat.each_row() %= trans((cgamma(cI3) - calpha) / cgamma(cI3));
			mu.cols(cI3) = tmp_mat;
		}
		u = mu.each_col() + rho;
		v = norm2kjv(K, u);
		wj = norm2kjw(K, u, v);
		tmp = wj%cgamma;
		qm = prox(tmp, C, cgamma);
		activeset = find(qm > 0);
		if (activeset.n_elem == 0 || fabs((primalobj - mod_dualobj) / primalobj) < tol) {
			break;
		}
		if (l == OuterMaxIter) {
            Rcout << "Does not converge in outer cycle." << endl;
		}
	}
	work = zeros(M);
	if (activeset.n_elem > 0) {
		work(activeset) = qm(activeset) / wj(activeset);
	}
	else {
		cb = -(log(.5* (mean(yapp) + 1)) / log(1 - .5* (mean(yapp) + 1)));
	}
	d = work / (1 - work / cgamma);
	sumd = sum(d);
	if (sumd != 0) {
		d /= sumd;
	}
	res["alpha"] = -sumd*rho*d.t();
	res["b"] = -cb;
	res["weight"] = d;
	res["rho"] = -sumd*rho;
	return res;
}


//[[Rcpp::export]]
arma::vec predictspicy(arma::mat alpha, double b, arma::cube k0) {
	int mm = k0.n_rows, pp = k0.n_slices;
	vec yy = b*ones(mm);
	for (int i = 0; i < pp; i++) {
		yy += k0.slice(i)*alpha.col(i);
	}
	return yy;
}


//[[Rcpp::depends(RcppArmadillo)]]
double clambda1, clambda, lamlam1;
int n, m;
arma::cube k;
uvec actset, yrisk;
arma::vec delta, rho, y;

double knorm(arma::vec v, arma::mat k_slice_m) {
	return sqrt(dot(k_slice_m*v, v));
}

double dg(double yy) {
	return (yy - clambda1) / clambda / yy;
}

arma::vec invcumsum(int nn, arma::vec x) {
	arma::vec cx(nn);
	double init = 0;
	vec::iterator x_begin = x.begin(), x_end = x.end(), cx_end = cx.end();
	while (x_end != x_begin) {
		init += *(--x_end);
		*(--cx_end) = init;
	}
	return cx;
}


inline int tsign(double ti, int ci, double tj, int cj) {
	if (ti > tj) {
		return cj;
	}
	else if (ti < tj) {
		return -ci;
	}
	else {
		return cj - ci;
	}
}

inline int getsign(double x) {
	if (x > 0) {
		return 1;
	}
	else if (x < 0) {
		return -1;
	}
	else {
		return 0;
	}
}

double pl(int n, arma::vec score, arma::vec del) {
	double mi, mi1;
	arma::vec escore = exp(score), yscr = log(invcumsum(n, escore));
	if (!is_finite(yscr)) {
		uvec nonfiidx = find_nonfinite(yscr);
		bool pos = any(yscr(nonfiidx) > 0), neg = any(yscr(nonfiidx) < 0);
		unsigned np;
		arma::vec subscore, subscore2;
		if (pos) {
			np = sum(yscr(nonfiidx) > 0);
			mi = max(score);
			subscore = score - mi;
			yscr(0) = mi + log(sum(exp(subscore)));
			if (np > 1) {
				for (unsigned i = 1; i < np; ++i) {
					mi1 = max(score.tail(n - i));
					subscore += mi - mi1;
					mi = mi1;
					yscr(i) = mi + log(sum(exp(subscore.tail(n - i))));
				}
			}
		}
		if (neg) {
			np = sum(yscr(nonfiidx) < 0);
			mi = min(score.tail(np));
			subscore = score.tail(np) - mi;
			yscr(n - np) = mi + log(sum(exp(subscore)));
			if (np > 1) {
				for (unsigned i = n - np + 1; i < (unsigned)n; i++) {
					mi1 = min(score.tail(n - i));
					subscore += mi - mi1;
					mi = mi1;
					yscr(i) = mi + log(sum(exp(subscore.tail(n - i))));
				}
			}
		}
	}
	return dot(del, score - yscr);
}

double cidx(int n, arma::vec score, arma::vec& time, arma::vec& del) {
	int up = 0, down = 0, sct;
	vec::iterator t1 = time.begin() + 1, te = time.end(), d1 = del.begin() + 1, s1 = score.begin() + 1, t2, d2, s2;
	while (t1 != te) {
		t2 = time.begin();
		d2 = del.begin();
		s2 = score.begin();
		while (t2 != t1) {
			sct = tsign(*t1, *d1, *t2++, *d2++);
			up += sct*getsign(*s2++ - *s1);
			down += fabs(sct);
		}
		++t1;
		++d1;
		++s1;
	}
	return .5*((double)up / down + 1);
}

arma::vec graCox() {
	arma::vec ggrho, rhoc0, rhoc, delrhoc, rhocsum, delrhocsum;
	double rhok;
	rhoc0 = -invcumsum(n - 1, rho.tail(n - 1));
	rhoc = log(rhoc0);
	delrhoc = log(delta.tail(n - 1) + rhoc0);
	rhocsum = invcumsum(n - 1, rhoc);
	delrhocsum = invcumsum(n - 1, delrhoc);
	rhocsum.insert_rows(n - 1, zeros<vec>(1));
	delrhocsum.insert_rows(n - 1, zeros<vec>(1));
	ggrho = rhocsum - delrhocsum - log(delta - rho) + 1e5*sum(rho)*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			if (rhok > clambda1) {
				ggrho += dg(rhok)*k.slice(i)*rho;
			}
			else {
				actset(i) = 0;
			}
		}
	}
	return ggrho;
}

arma::mat hesCox() {
	arma::mat hesrr = diagmat(1 / (delta - rho)) + 1e5*ones(n, n);
	double rhok;
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			hesrr += (rhok - clambda1) / clambda / rhok*k.slice(i) + lamlam1 * pow(rhok, -3)*k.slice(i)*rho*trans(rho)*k.slice(i);
		}
	}
	return hesrr + eye(n, n)*1e-8;
}




//[[Rcpp::export]]
arma::mat Coxdual(arma::vec y0, arma::vec delta0, arma::cube k0, arma::vec rho0, double cc, double lambda, int maxiter, double cri) {
	k = k0;
	rho = rho0;
	n = delta0.n_elem;
	m = k.n_slices;
	delta = delta0;
	clambda = cc*lambda;
	clambda1 = cc*(1 - lambda);
	lamlam1 = clambda1 / clambda;
	actset = ones<uvec>(m);
	y = y0;
	int i;
	double rhok, st = 1, s1, s2;
	arma::vec g, rhonew, hg, arhonew, ahg(n - 1), arho(n - 1);
	arma::mat h, alpha = zeros(n, m);
	uvec yxd0, rdd0;
	for (i = 0; i < maxiter; i++) {
		g = graCox();
		h = hesCox();
		hg = -solve(h, g);
		rhonew = rho + hg;
		ahg = invcumsum(n - 1, hg.tail(n - 1));
		arho = invcumsum(n - 1, rho.tail(n - 1));
		arhonew = arho + ahg;
		if (any(arhonew >= 0) || any(rhonew >= delta)) {
			yxd0 = find(ahg > 0);
			rdd0 = find(hg > 0);
			s1 = yxd0.n_elem > 0 ? (min(-arho(yxd0) / ahg(yxd0))*.999) : 1;
			s2 = rdd0.n_elem > 0 ? (min((delta(rdd0) - rho(rdd0)) / hg(rdd0))*.999) : 1;
			st = s1 > s2 ? s2 : s1;
		}
		rhonew = rho + st*hg;
		if (all(actset == 0) || (norm(rho - rhonew) / norm(rho) < cri)) {
			break;
		}
		rho = rhonew;
		st = 1;
	}
	for (i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			alpha.col(i) = dg(rhok)*rho;
		}
	}
	return alpha;
}

//[[Rcpp::export]]
arma::vec predictsurv(arma::mat alpha, arma::cube k0) {
	int mm = k0.n_cols, pp = k0.n_slices;
	arma::vec yy = zeros(mm);
	for (int i = 0; i < pp; i++) {
		yy += k0.slice(i).t()*alpha.col(i);
	}
	return yy;
}




arma::mat kercom(arma::cube xx, arma::vec weight) {
	int m = xx.n_slices, n = xx.n_rows;
	arma::mat res = zeros(n, n);
	for (int i = 0; i < m; i++) {
		res += weight(i)*xx.slice(i);
	}
	return res;
}


arma::cube mixkercd(arma::mat xc, arma::mat xd) {
	int n = xc.n_rows, dc = xc.n_cols;
	rowvec mm = max(xc) - min(xc);
	arma::cube kc(n, n, 5);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = exp(-norm(xc.row(i) - xc.row(j)) / dc);
			kc(i, j, 0) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 1) = exp(-pow(norm(xc.row(i) - xc.row(j)), 2) / dc);
			kc(i, j, 1) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 1) = kc(i, j, 1);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 2) = dc - sum(abs(xc.row(i) - xc.row(j)) / mm);
			kc(i, j, 2) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 2) = kc(i, j, 2);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 3) = dot(xc.row(i), xc.row(j));
			kc(i, j, 3) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 3) = kc(i, j, 3);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 4) = pow(dot(xc.row(i), xc.row(j)) / dc, 3);
			kc(i, j, 4) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 4) = kc(i, j, 4);
			}
		}
	}
	return kc;
}



arma::cube mixkertestcd(arma::mat trc, arma::mat trd, arma::mat tec, arma::mat ted) {
	int ntr = trc.n_rows, nte = tec.n_rows, dc = trc.n_cols;
	rowvec mm = max(trc) - min(trc);
	arma::cube kc(ntr, nte, 5);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = exp(-norm(trc.row(i) - tec.row(j)) / dc);
			kc(i, j, 0) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 1) = exp(-pow(norm(trc.row(i) - tec.row(j)), 2) / dc);
			kc(i, j, 1) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 2) = dc - sum(abs(trc.row(i) - tec.row(j)) / mm);
			kc(i, j, 2) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 3) = dot(trc.row(i), tec.row(j));
			kc(i, j, 3) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 4) = pow(dot(trc.row(i), tec.row(j)) / dc, 3);
			kc(i, j, 4) += sum(trd.row(i) == ted.row(j));
		}
	}
	return kc;
}


arma::cube mixkerc(arma::mat xc) {
	int n = xc.n_rows, dc = xc.n_cols;
	rowvec mm = max(xc) - min(xc);
	arma::cube kc(n, n, 5);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = exp(-norm(xc.row(i) - xc.row(j)) / dc);
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 1) = exp(-pow(norm(xc.row(i) - xc.row(j)), 2) / dc);
			if (j > i) {
				kc(j, i, 1) = kc(i, j, 1);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 2) = dc - sum(abs(xc.row(i) - xc.row(j)) / mm);
			if (j > i) {
				kc(j, i, 2) = kc(i, j, 2);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 3) = dot(xc.row(i), xc.row(j));
			if (j > i) {
				kc(j, i, 3) = kc(i, j, 3);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 4) = pow(dot(xc.row(i), xc.row(j)) / dc, 3);
			if (j > i) {
				kc(j, i, 4) = kc(i, j, 4);
			}
		}
	}
	return kc;
}



arma::cube mixkertestc(arma::mat trc, arma::mat tec) {
	int ntr = trc.n_rows, nte = tec.n_rows, dc = trc.n_cols;
	rowvec mm = max(trc) - min(trc);
	arma::cube kc(ntr, nte, 5);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = exp(-norm(trc.row(i) - tec.row(j)) / dc);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 1) = exp(-pow(norm(trc.row(i) - tec.row(j)), 2) / dc);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 2) = dc - sum(abs(trc.row(i) - tec.row(j)) / mm);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 3) = dot(trc.row(i), tec.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 4) = pow(dot(trc.row(i), tec.row(j)) / dc, 3);
		}
	}
	return kc;
}


arma::cube mixkerd(arma::mat xd) {
	int n = xd.n_rows;
	arma::cube kc(n, n, 1);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	return kc;
}


arma::cube mixkertestd(arma::mat trd, arma::mat ted) {
	int ntr = trd.n_rows, nte = ted.n_rows;
	arma::cube kc(ntr, nte, 1);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = sum(trd.row(i) == ted.row(j));
		}
	}
	return kc;
}




arma::cube cvcoxmkl(const arma::vec& yy, const arma::mat& xx, const arma::vec& del, vec ccsearch, arma::vec lamsearch, const uvec& incd, const uvec& cvwhich, int nf, int maxiter, double cri, bool meas) {
	int ncc = ccsearch.n_elem, nll = lamsearch.n_elem, ntr, nva = 0, nfull = yy.n_elem;
	arma::vec yytr, rho0, ytr, yva, scoreva, ddeltr, deltr, delva;
	arma::mat xxtr, xtr, xva, modelre;
	uvec trind, vaind, indic = find(incd == 1), incon = find(incd == 0), yf1, yz1, ycom, trsrt;
	arma::cube ktr, kva, kfull, cvppl(ncc, nll, nf);
	for (int i = 0; i < nf; i++) {
		trind = find(cvwhich != i);
		ntr = trind.n_elem;
		xxtr = xx.rows(trind);
		yytr = yy.elem(trind);
		ddeltr = del.elem(trind);
		trsrt = sort_index(yytr);
		xtr = xxtr.rows(trsrt);
		ytr = sort(yytr);
		deltr = ddeltr.elem(trsrt);
		rho0 = .001*(deltr - linspace(0, 10, ntr));
		arma::mat yortr(ntr, ntr);
		if (meas == 0) {
			if (indic.n_elem == 0) {
				ktr = mixkerc(xtr);
				kfull = mixkertestc(xtr, xx);
			}
			else if (incon.n_elem == 0) {
				ktr = mixkerd(xtr);
				kfull = mixkertestd(xtr, xx);
			}
			else {
				ktr = mixkercd(xtr.cols(incon), xtr.cols(indic));
				kfull = mixkertestcd(xtr.cols(incon), xtr.cols(indic), xx.cols(incon), xx.cols(indic));
			}
		}
		else {
			vaind = find(cvwhich == i);
			nva = vaind.n_elem;
			xva = xx.rows(vaind);
			yva = yy.elem(vaind);
			delva = del.elem(vaind);
			if (indic.n_elem == 0) {
				ktr = mixkerc(xtr);
				kva = mixkertestc(xtr, xva);
			}
			else if (incon.n_elem == 0) {
				ktr = mixkerd(xtr);
				kva = mixkertestd(xtr, xva);
			}
			else {
				ktr = mixkercd(xtr.cols(incon), xtr.cols(indic));
				kva = mixkertestcd(xtr.cols(incon), xtr.cols(indic), xva.cols(incon), xva.cols(indic));
			}
		}
		for (int j = 0; j < ncc; j++) {
			for (int h = 0; h < nll; h++) {
				modelre = Coxdual(ytr, deltr, ktr, rho0, ccsearch(j), lamsearch(h), maxiter, cri);
				if (meas) {
					cvppl(j, h, i) = pl(nfull, predictsurv(modelre, kfull), del) - pl(ntr, predictsurv(modelre, ktr), deltr);
				}
				else {
					scoreva = -predictsurv(modelre, kva);
					cvppl(j, h, i) = cidx(nva, predictsurv(modelre, kva), yva, delva);
				}
			}
		}
	}
	return cvppl;
}

