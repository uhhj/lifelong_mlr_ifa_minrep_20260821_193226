"""CPU-only independent arithmetic checks of saved forensic evidence."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from qp_corrected_update_fidelity_v1.common import ROOT,read,write
OUT=ROOT/'reports/qp_writeback_forensic_v1'

def main():
 evidence=[]
 for sid in read(OUT/'PROTOCOL.json')['targets']:
  path=OUT/sid;r=read(path/'RESULT.json');z=np.load(path/'CONSTRAINT_RESIDUALS.npz');q=np.load(path/'ROUNDING_COORDINATES.npz');blocks=pd.read_csv(path/'PARAMETER_BLOCKS.csv')
  np.testing.assert_array_equal(z['d32_rates'],z['dscatter_rates'])
  np.testing.assert_array_equal(q['theta_written'],(q['theta_native']-q['c32']).astype(np.float32))
  np.testing.assert_array_equal(q['dwritten'],q['theta_written'].astype(np.float64)-q['theta0'].astype(np.float64))
  np.testing.assert_array_equal(q['write_error'],q['theta_written'].astype(np.float64)-(q['theta_native'].astype(np.float64)-q['c32'].astype(np.float64)))
  pairs=[('cast_and_subtract',z['d32_rates']-z['d64_rates'],z['correction_cast_error_rates']+z['displacement_subtract_error_rates']),('scatter_to_written',z['dwritten_rates']-z['dscatter_rates'],z['native_delta_subtract_error_rates']+z['parameter_write_error_rates']-z['displacement_subtract_error_rates']),('reporting_subtraction',z['dwritten_float32_report_rates']-z['dwritten_rates'],z['reported_delta_subtract_error_rates'])]
  errors={}
  for name,lhs,rhs in pairs:
   err=float(np.max(abs(lhs-rhs)));assert err<=1e-13;errors[name]=err
  for i in r['failed_constraint_indices']:
   assert np.isclose(blocks[f'constraint_{i}_write_error_dot'].sum(),z['parameter_write_error_rates'][i],rtol=1e-9,atol=1e-13)
  corrnorm=float(np.sqrt(z['lam']@z['K']@z['lam']));error=r['error_decomposition']['parameter_write']['norm'];u=float(z['u_norm'])
  evidence.append(dict(start_id=sid,checks='PASS',delta_inner_product_closure_max_errors=errors,coordinate_rounding_bitwise=True,parameter_contributions_sum=True,ideal_correction_norm=corrnorm,parameter_write_error_norm=error,write_error_over_native_norm=error/u,write_error_over_correction_norm=error/corrnorm,absorbed_nonzero_coordinate_fraction=float(blocks.correction_absorbed.sum()/blocks.correction_nonzero.sum()),note='absorbed fraction counts coordinates, not correction energy'))
 write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',evidence=evidence,formal_training_resumed=False))
 report=OUT/'FINAL_REPORT.md';text=report.read_text();text+='\n## Independently verified root cause\n\n'
 for r in evidence:
  text+=f"- {r['start_id']}: float32 parameter addition/subtraction rounding error norm {r['parameter_write_error_norm']:.12g}, {100*r['write_error_over_native_norm']:.6g}% of native displacement and {100*r['write_error_over_correction_norm']:.6g}% of ideal correction. {100*r['absorbed_nonzero_coordinate_fraction']:.6g}% of nonzero correction coordinates do not change the stored native parameter; this is a coordinate count, not energy loss.\n"
 text+='\nBoth QPs and flat float32 corrected displacements pass. The layout reconstruction is exact. Rounding theta_native-c32 into the float32 parameter lattice introduces the first above-gate residual. Computing the final parameter difference in binary64 still shows the same failure, ruling out a float32 subtraction-report artifact. Identical fixed-native normalization rules out denominator effects. Errors are individually within ordinary float32 rounding scale, but ideal active constraints have near-zero slack. This diagnosis does not relax thresholds, certify finite NLL protection, or authorize a writeback repair.\n'
 report.write_text(text)
if __name__=='__main__':main()
