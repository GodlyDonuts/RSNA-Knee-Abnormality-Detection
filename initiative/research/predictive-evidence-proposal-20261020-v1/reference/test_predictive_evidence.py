"""Synthetic mathematical tests only. No MRI, radiology labels, or AUC claims."""
import json
import math
import unittest
from pathlib import Path

import torch
from predictive_evidence import predictive_evidence, weak_supervised_loss, average_mask_posteriors


def fixture():
    g = torch.Generator().manual_seed(20261020)
    B, C, J, D = 3, 2, 5, 4
    t = torch.randn(B, J, D, generator=g, dtype=torch.float64)
    m = torch.randn(B, C, J, D, generator=g, dtype=torch.float64)
    d = torch.randn(B, C, J, D, generator=g, dtype=torch.float64) * .3
    v = torch.randn(C, D, generator=g, dtype=torch.float64) * .1
    s = torch.eye(J, dtype=torch.bool)[None].expand(C, -1, -1).clone()
    w = torch.zeros(C, J, dtype=torch.float64)
    o = torch.tensor([[1,1,0,0,1],[1,0,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    return t,m,d,v,s,w,o


class Tests(unittest.TestCase):
    def test_exact_gaussian_mixture(self):
        args = fixture(); t,m,d,v,s,w,o = args
        out = predictive_evidence(*args)
        rows=[]
        for b in range(len(t)):
            cs=[]
            for c in range(m.shape[1]):
                base = torch.distributions.Normal(m[b,c,o[b]], (v[c]/2).exp())
                lp0 = base.log_prob(t[b,o[b]]).sum()
                terms=[]
                for k in range(s.shape[1]):
                    mean=m[b,c] + d[b,c]*s[c,k,:,None]
                    dist=torch.distributions.Normal(mean[o[b]], (v[c]/2).exp())
                    terms.append(dist.log_prob(t[b,o[b]]).sum()+w[c].log_softmax(0)[k])
                cs.append(torch.stack(terms).logsumexp(0)-lp0)
            rows.append(torch.stack(cs))
        torch.testing.assert_close(out.log_bayes_factor,torch.stack(rows),atol=1e-12,rtol=1e-12)

    def test_no_evidence_is_neutral(self):
        a=list(fixture());a[-1]=torch.zeros_like(a[-1]);out=predictive_evidence(*a)
        torch.testing.assert_close(out.log_bayes_factor,torch.zeros_like(out.log_bayes_factor))
        torch.testing.assert_close(out.no_observation_prior_mass,torch.ones_like(out.no_observation_prior_mass))

    def test_identical_hypotheses_neutral(self):
        a=list(fixture());a[2]=torch.zeros_like(a[2]);out=predictive_evidence(*a)
        torch.testing.assert_close(out.log_bayes_factor,torch.zeros_like(out.log_bayes_factor))

    def test_missing_support_bound(self):
        out=predictive_evidence(*fixture());mask=out.no_observation_prior_mass>0
        self.assertTrue(torch.all(out.log_bayes_factor[mask] >= out.no_observation_prior_mass[mask].log()-1e-12))

    def test_unobserved_values_irrelevant(self):
        a=list(fixture());ref=predictive_evidence(*a);a[0]=a[0].masked_fill(~a[-1][...,None],float('nan'))
        got=predictive_evidence(*a);torch.testing.assert_close(ref.log_bayes_factor,got.log_bayes_factor)

    def test_region_permutation_invariant(self):
        a=list(fixture());ref=predictive_evidence(*a);p=torch.tensor([4,1,3,0,2])
        a[0]=a[0][:,p];a[1]=a[1][:,:,p];a[2]=a[2][:,:,p];a[4]=a[4][:,:,p];a[6]=a[6][:,p]
        torch.testing.assert_close(ref.log_bayes_factor,predictive_evidence(*a).log_bayes_factor)

    def test_duplicate_support_mass_split_invariant(self):
        a=list(fixture());ref=predictive_evidence(*a)
        a[4]=a[4].repeat_interleave(2,1);a[5]=a[5].repeat_interleave(2,1)-math.log(2)
        torch.testing.assert_close(ref.log_bayes_factor,predictive_evidence(*a).log_bayes_factor)

    def test_posteriors_normalized(self):
        out=predictive_evidence(*fixture())
        torch.testing.assert_close(out.support_log_posterior.exp().sum(-1),torch.ones_like(out.log_bayes_factor))

    def test_zero_shift_still_has_gradient(self):
        a=list(fixture());a[2]=torch.zeros_like(a[2],requires_grad=True)
        predictive_evidence(*a).log_bayes_factor.sum().backward()
        self.assertGreater(float(a[2].grad.abs().sum()),0)

    def test_autograd_gradcheck(self):
        a=list(fixture());a[2]=a[2].requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda d: predictive_evidence(a[0],a[1],d,*a[3:]).log_bayes_factor,
                                                (a[2],), eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_common_nuisance_shift_cancels(self):
        a=list(fixture());ref=predictive_evidence(*a);s=torch.full_like(a[0],20)
        a[0]=a[0]+s;a[1]=a[1]+s[:,None]
        torch.testing.assert_close(ref.log_bayes_factor,predictive_evidence(*a).log_bayes_factor)

    def test_hypothesis_prior_matters(self):
        a=list(fixture());ref=predictive_evidence(*a);a[5][:,0]=10
        self.assertFalse(torch.allclose(ref.log_bayes_factor,predictive_evidence(*a).log_bayes_factor))

    def test_reject_observed_nan(self):
        a=list(fixture());a[0][0,0,0]=float('nan')
        with self.assertRaises(ValueError):predictive_evidence(*a)

    def test_reject_collapsing_target(self):
        a=list(fixture());a[0]=a[0].requires_grad_()
        with self.assertRaises(ValueError):predictive_evidence(*a)

    def test_reject_half_precision(self):
        a=[v.half() if v.is_floating_point() else v for v in fixture()]
        with self.assertRaises(ValueError):predictive_evidence(*a)

    def test_all_masked_source_loss_zero(self):
        a=fixture();out=predictive_evidence(*a);z=torch.zeros_like(out.log_bayes_factor)
        loss,_=weak_supervised_loss(out,z,torch.full_like(z,float('nan')),torch.zeros_like(z,dtype=torch.bool),a[-1],4)
        self.assertEqual(float(loss),0)

    def test_soft_source_loss_finite(self):
        a=fixture();out=predictive_evidence(*a);z=torch.zeros_like(out.log_bayes_factor)
        loss,_=weak_supervised_loss(out,z,torch.full_like(z,.2),torch.ones_like(z,dtype=torch.bool),a[-1],4)
        self.assertTrue(torch.isfinite(loss))

    def test_average_posteriors_not_logit_sum(self):
        a=torch.tensor([[[2.],[-1.]]]); b=torch.tensor([[[.7],[.5]]])
        torch.testing.assert_close(average_mask_posteriors(a,b),(a+b).sigmoid().mean(1))
        self.assertFalse(torch.allclose(average_mask_posteriors(a,b),(a+b).sum(1).sigmoid()))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "successful":result.wasSuccessful(),"torch_version":torch.__version__,"device":"CPU",
             "MRI_data_used":False,"clinical_labels_used":False,"MRI_training_completed":False,
             "accuracy_gain_claim":False,"scope":"mathematical likelihood and gradient tests on synthetic tensors"}
    Path(__file__).with_name('test_results.json').write_text(json.dumps(payload,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
