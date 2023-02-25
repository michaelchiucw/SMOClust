package moa.streams.filters;

import java.util.LinkedList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

public class BinaryToNominalFilter extends AbstractStreamFilter {
	
    public StringOption binToNomIndicesOption = new StringOption("binToNomIndices", 'b',
    		"Determine which binary attributes to merged as a nominal attribute.\n" + 
    		"The range should be integers seperated by hypen (-). e.g. \"2-4\"\n" + 
    		"If there are multiple lists of binary attributes to be merged as nominal attributes, please seperate them with semi-coloms (;).",
    		"");
    
    public StringOption nomNamesOption = new StringOption("nomNames", 'n',
    		"Name(s) for the nominal attribute(s) after merging the binary attributes.\n" + 
    		"If there are multiple names for different nominal attributes, please seperate them with semi-coloms (;). e.g. \"nom_attr1;nom_attr2\"",
    		"");
    
    protected int[][] binaryAttributeIndices;
    protected String[] nomNames;
    protected Instances converted_header;

	@Override
	public InstancesHeader getHeader() {
		if (this.converted_header == null) {
			Instances original_bin_header = this.inputStream.getHeader();
			
			List<Attribute> converted_attributes = new LinkedList<Attribute>();
			
			int binaryAttributeIndices_iterator = 0;
			List<String> possible_values = new LinkedList<String>();
			
			for (int i = 0; i < original_bin_header.numAttributes(); ++i) {
				
				if (i >= binaryAttributeIndices[binaryAttributeIndices_iterator][0] && i <= binaryAttributeIndices[binaryAttributeIndices_iterator][1]) {
					possible_values.add(original_bin_header.attribute(i).name());
					
					if (i == binaryAttributeIndices[binaryAttributeIndices_iterator][1]) {
						Attribute newNomAttr = new Attribute(nomNames[binaryAttributeIndices_iterator], possible_values);
						converted_attributes.add(newNomAttr);
						possible_values = new LinkedList<String>();
						if (binaryAttributeIndices_iterator < binaryAttributeIndices.length - 1) {
							++binaryAttributeIndices_iterator;
						}
					}
					continue;
				}
				converted_attributes.add(original_bin_header.attribute(i));
			}
			
			this.converted_header = new Instances(original_bin_header.getRelationName(), converted_attributes, 1);
			String classAttribute_name = original_bin_header.classAttribute().name();
			for (int i = 0; i < this.converted_header.numAttributes(); ++i) {
				if (this.converted_header.attribute(i).name().equals(classAttribute_name)) {
					this.converted_header.setClassIndex(i);
				}
			}
			
			this.converted_header.setRelationName(original_bin_header.getRelationName() + "-moa.streams.filters.BinaryToNominalFilter");
			
		}
		return new InstancesHeader(this.converted_header);
	}
	
	@Override
    public String getPurposeString() {
		return "Convert binary (dummy) attributes to nominal attributes.";
    }
	
	@Override
	public void getDescription(StringBuilder sb, int indent) {
		
	}

	@Override
	protected void restartImpl() {
		String[] lists_of_binaryAttributes = this.binToNomIndicesOption.getValue().split(";");
		this.binaryAttributeIndices = new int[lists_of_binaryAttributes.length][];
		for (int i = 0; i < this.binaryAttributeIndices.length; ++i) {
			this.binaryAttributeIndices[i] = Arrays.stream(lists_of_binaryAttributes[i].split("-")).mapToInt(Integer::parseInt).toArray();
		}
		
		this.nomNames = this.nomNamesOption.getValue().split(";");
	}
	
	@Override
    public Instance filterInstance(Instance inst) {
		Instance filteredInst = new DenseInstance(this.converted_header.numAttributes());
		filteredInst.setDataset(this.converted_header);
		
		int binaryAttributeIndices_iterator = 0;
		int filteredInst_attribute_iterator = 0;
		List<Double> values = new LinkedList<Double>();
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (i >= binaryAttributeIndices[binaryAttributeIndices_iterator][0] && i <= binaryAttributeIndices[binaryAttributeIndices_iterator][1]) {
				// Nominal Attribute
				values.add(inst.value(i));
				if (i == binaryAttributeIndices[binaryAttributeIndices_iterator][1]) {
					double maxValue = Collections.max(values);
					int maxIndex = values.indexOf(maxValue);
					filteredInst.setValue(filteredInst_attribute_iterator++, maxIndex);
					values = new LinkedList<Double>();
					if (binaryAttributeIndices_iterator < binaryAttributeIndices.length - 1) {
						++binaryAttributeIndices_iterator;
					}
				}
			} else {
				// Numeric Attribute
				filteredInst.setValue(filteredInst_attribute_iterator++, inst.value(i));
			}
		}
		return filteredInst;
    }
	
}