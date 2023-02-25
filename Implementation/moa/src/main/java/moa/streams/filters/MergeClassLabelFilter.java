package moa.streams.filters;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

public class MergeClassLabelFilter extends AbstractStreamFilter {
	
	public StringOption classToMergeOption = new StringOption("classToMerge", 'c',
    		"Determine which classes to merge.\n" +
    		"The set of class indices to be merged as one should be integers seperated by comma (,). e.g. \"1,2,4\"\n" + 
    		"If there are multiple lists of class to be merged as one, please seperate them with semi-coloms (;). \n" +
    		"e.g. \"1,2,4;3,5,6\"",
    		"");
	
	protected int[][] classToMerge;
	protected Instances converted_header;
	
	@Override
	public InstancesHeader getHeader() {
		if (this.converted_header == null) {
			Instances original_header = this.inputStream.getHeader();
			// Create a new header based on the current header and modify the relation name.
			this.converted_header = new Instances(original_header);
			this.converted_header.setRelationName(original_header.getRelationName() + "-moa.streams.filters.CombineClassLabelFilter");
			
			Attribute original_class_attribute = original_header.classAttribute();
			List<String> original_class_values = original_class_attribute.getAttributeValues();
			List<String> new_class_values = new LinkedList<String>();
			// Add possible class values that are not going to be merged to the new List.
			for (int i = 0; i < original_class_values.size(); ++i) {
				boolean toAdd = true;
				for (int[] class_values_arr : this.classToMerge) {
					for (int class_value : class_values_arr) {
						if (class_value == i) {
							toAdd = false;
						}
					}
				}
				if (toAdd) {
					new_class_values.add(original_class_values.get(i));
				}
			}
			// Add merged possible values
			for (int[] classValuesToMerge : this.classToMerge) {
				StringBuilder mergedClassValue_builder = new StringBuilder();
				for (int i = 0; i < classValuesToMerge.length; ++i) {
					mergedClassValue_builder.append(original_class_values.get(classValuesToMerge[i]));
					if (i < classValuesToMerge.length - 1) {
						mergedClassValue_builder.append("-");
					}
				}
				new_class_values.add(mergedClassValue_builder.toString());
			}
			Collections.sort(new_class_values);
			
			Attribute new_class_attribute = new Attribute(original_class_attribute.name(), new_class_values);
			
			// Replace old class attribute with the new one.
			int target_class_index = this.converted_header.classIndex();
			this.converted_header.insertAttributeAt(new_class_attribute, this.converted_header.classIndex());
			this.converted_header.deleteAttributeAt(this.converted_header.classIndex());
			this.converted_header.setClassIndex(target_class_index);
			
		}
		
		return new InstancesHeader(this.converted_header);
	}
	
	@Override
    public String getPurposeString() {
		return "Combine classes into one class.";
    }

	@Override
	public void getDescription(StringBuilder sb, int indent) {

	}

	@Override
	protected void restartImpl() {
		String[] classToCombine_str = this.classToMergeOption.getValue().split(";");
		this.classToMerge = new int[classToCombine_str.length][];
		for (int i = 0; i < this.classToMerge.length; ++i) {
			this.classToMerge[i] = Arrays.stream(classToCombine_str[i].split(",")).mapToInt(Integer::parseInt).toArray();
		}
	}
	
	@Override
    public Instance filterInstance(Instance inst) {
		String original_class_value_str = inst.classAttribute().value((int) inst.classValue());
		
		Instance filteredInst = new DenseInstance(this.converted_header.numAttributes());
		filteredInst.setDataset(this.converted_header);
		
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (inst.attribute(i).equals(inst.classAttribute())) {
				// Set class attribute
				List<String> new_class_values = this.converted_header.classAttribute().getAttributeValues();
				for (int j = 0; j < new_class_values.size(); ++j) {
					if (new_class_values.get(j).contains(original_class_value_str)) { // Use the actual class string to match.
						filteredInst.setClassValue((double) j);
						break;
					}
				}
			} else {
				// Set other attributes
				filteredInst.setValue(i, inst.value(i));
			}
		}
		
		return filteredInst;
	}

}
