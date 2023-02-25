/*
 *    SplitARFFinto2Files.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import moa.core.ObjectRepository;
import moa.options.ClassOption;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;

/**
 * Split an ARFF file into 2 files by percentage.
 *
 * @author Chun Wai Chiu (cxc1015@student.bham.ac.uk)
 * @version $Revision: 1 $
 */
public class SplitARFFinto2Files extends AuxiliarMainTask {

    @Override
    public String getPurposeString() {
        return "Split an ARFF file into 2 files by percentage.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to write.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public FileOption arffFile1Option = new FileOption("arffFile1", 'a',
            "Destination ARFF file 1.", null, "arff", true);
    
    public FileOption arffFile2Option = new FileOption("arffFile2", 'b',
            "Destination ARFF file 2.", null, "arff", true);

    public IntOption maxInstancesOption = new IntOption("maxInstances", 'm',
            "Maximum number of instances of the original ARFF file.", 10000000, 0,
            Integer.MAX_VALUE);
    
    public FloatOption percentageOfSplitOption = new FloatOption("percentageOfSplit", 'p',
            "Percentage of the split.\n" +
            "e.g. if enter 0.1, the first 10% of the examples in the original ARFF file will go to file 1" +
            " and the rest of the examples will go to file 2", 0.1, 0,
            1);

    public FlagOption suppressHeaderOption = new FlagOption("suppressHeader",
            'h', "Suppress header from output.");

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
    	ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);
    	
    	File destFile1 = this.arffFile1Option.getFile();
    	File destFile2 = this.arffFile2Option.getFile();
    	
    	double total_examples_original = this.maxInstancesOption.getValue();
    	double splitPercentage = this.percentageOfSplitOption.getValue();
    	double first_file_examples_total = Math.floor(total_examples_original * splitPercentage);
    	double second_file_examples_total = total_examples_original - first_file_examples_total;
    	
    	if (destFile1 != null && destFile2 != null) {
    		try {
                Writer w = new BufferedWriter(new FileWriter(destFile1));
                monitor.setCurrentActivityDescription("Writing stream to ARFF");
                if (!this.suppressHeaderOption.isSet()) {
                    w.write(stream.getHeader().toString());
                    w.write("\n");
                }
                int numWritten = 0;
                while ((numWritten < first_file_examples_total)
                        && stream.hasMoreInstances()) {
                    w.write(stream.nextInstance().getData().toString());
                    w.write("\n");
                    numWritten++;
                }
                w.close();
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Failed writing to file " + destFile1, ex);
            }
            
            try {
                Writer w = new BufferedWriter(new FileWriter(destFile2));
                monitor.setCurrentActivityDescription("Writing stream to ARFF");
                if (!this.suppressHeaderOption.isSet()) {
                    w.write(stream.getHeader().toString());
                    w.write("\n");
                }
                int numWritten = 0;
                while ((numWritten < second_file_examples_total)
                        && stream.hasMoreInstances()) {
                    w.write(stream.nextInstance().getData().toString());
                    w.write("\n");
                    numWritten++;
                }
                w.close();
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Failed writing to file " + destFile2, ex);
            }
            return "Stream written to ARFF file " + destFile1 + "\nStream written to ARFF file " + destFile2;
    	}
        throw new IllegalArgumentException("No destination file to write to.");
    }

    @Override
    public Class<?> getTaskResultType() {
        return String.class;
    }
}
