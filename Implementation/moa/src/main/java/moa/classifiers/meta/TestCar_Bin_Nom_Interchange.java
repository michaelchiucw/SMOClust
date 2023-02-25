package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Collections;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.core.Measurement;

public class TestCar_Bin_Nom_Interchange extends AbstractClassifier implements MultiClassClassifier {
	
	NaiveBayes baseLearner;
	
	protected SamoaToWekaInstanceConverter moaToWekaInstanceConverter;
	protected WekaToSamoaInstanceConverter wekaToMoaInstanceConverter;

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return this.baseLearner.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		// TODO Auto-generated method stub
		this.baseLearner = new NaiveBayes();
		this.baseLearner.resetLearning();
		
		this.moaToWekaInstanceConverter = new SamoaToWekaInstanceConverter();
		this.wekaToMoaInstanceConverter = new WekaToSamoaInstanceConverter();
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
		this.baseLearner.trainOnInstance(inst);
		
		
		Instance nom2Bin_inst = null;
		Instance bin2Nom_inst = null;
		try {
			nom2Bin_inst = nominalToBinary(inst);
			bin2Nom_inst = binaryToNominal(nom2Bin_inst, inst.dataset());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		System.out.println("\nnom2Bin: " + Arrays.toString(nom2Bin_inst.toDoubleArray()));
//		System.out.println("bin2nom: " + Arrays.toString(bin2Nom_inst.toDoubleArray()));
//		System.out.println("original: " + Arrays.toString(inst.toDoubleArray()));
//		
//		System.out.println("bin2nom classIndex: " + bin2Nom_inst.classIndex());
//		System.out.println("bin2nom classValue: " + bin2Nom_inst.classValue());
		for (int i = 0; i < inst.numAttributes(); ++i) {
			double ori_value = inst.value(i);
			double syn_value = bin2Nom_inst.value(i);
			if (ori_value != syn_value) {
				System.out.println("diff: " + ori_value + ", " + syn_value);
			}
		}
	}
	
	private Instance binaryToNominal(Instance nom_inst, Instances original_header) throws Exception {
		
		Instances bin2nom_insts = new Instances(original_header);
		Instance tmp_inst = new DenseInstance(original_header.numAttributes());
		tmp_inst.setDataset(original_header);
		
		for (int i = 0; i < bin2nom_insts.numAttributes(); ++i) {
			String current_attr_name = bin2nom_insts.attribute(i).name();
			ArrayList<Double> values = new ArrayList<Double>();
			
			for (int j = 0; j < nom_inst.numAttributes(); ++j) {
				if (nom_inst.attribute(j).name().contains(current_attr_name)) {
					values.add(nom_inst.value(j));
				}
			}
			
			if (values.size() == 1) { // Numeric attrbute
				tmp_inst.setValue(i, values.get(0));
			} else if (values.size() > 1) { // Binary attribute: to be converted to nominal attribute
				
				double maxValue = Collections.max(values);
				int maxIndex = values.indexOf(maxValue);
				tmp_inst.setValue(i, maxIndex);
				
			} else { // matched_attributes.size == 0, which shouldn't happen but just in case.
				throw new Exception("No matched attribute.");
			}
		}
		bin2nom_insts.add(tmp_inst);
		
		return bin2nom_insts.get(0);
	}
	
	private Instance nominalToBinary(Instance original_inst) throws Exception {
		
		Instances moaInstances = new Instances(original_inst.dataset());
		moaInstances.add(original_inst);
		
		weka.core.Instances wekaInstances = this.moaToWekaInstanceConverter.wekaInstances(moaInstances);
		weka.filters.unsupervised.attribute.NominalToBinary nom2BinFilter = new weka.filters.unsupervised.attribute.NominalToBinary();
		nom2BinFilter.setInputFormat(wekaInstances);
		wekaInstances = weka.filters.Filter.useFilter(wekaInstances, nom2BinFilter);
		moaInstances = this.wekaToMoaInstanceConverter.samoaInstances(wekaInstances);
		
		return moaInstances.get(0);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}

}
