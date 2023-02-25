/*
 *    SphereCluster.java
 *    Copyright (C) 2010 RWTH Aachen University, Germany
 *    @author Jansen (moa@cs.rwth-aachen.de)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 *    
 */

package moa.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * A simple implementation of the <code>Cluster</code> interface representing
 * spherical clusters. The inclusion probability is one inside the sphere and zero
 * everywhere else.
 *
 */
public class SphereCluster extends Cluster {

	private static final long serialVersionUID = 1L;

	private double[] center;
	private double radius;
	private double weight;


	public SphereCluster(double[] center, double radius) {
		this( center, radius, 1.0 );
	}

	public SphereCluster() {
	}

	public SphereCluster( double[] center, double radius, double weightedSize) {
		this();
		this.center = center;
		this.radius = radius;
		this.weight = weightedSize;
	}

	public SphereCluster(int dimensions, double radius, Random random) {
		this();
		this.center = new double[dimensions];
		this.radius = radius;

		// Position randomly but keep hypersphere inside the boundaries
		double interval = 1.0 - 2 * radius;
		for (int i = 0; i < center.length; i++) {
			this.center[i] = (random.nextDouble() * interval) + radius;
		}
		this.weight = 0.0;
	}


	public SphereCluster(List<?extends Instance> instances, int dimension){
		this();
		if(instances == null || instances.size() <= 0)
			return;

		weight = instances.size();

		Miniball mb = new Miniball(dimension);
		mb.clear();

		for (Instance instance : instances) {
			mb.check_in(instance.toDoubleArray());
		}

		mb.build();
		center = mb.center();
		radius = mb.radius();
		mb.clear();
	}


	/**
	 * Checks whether two <code>SphereCluster</code> overlap based on radius
	 * NOTE: overlapRadiusDegree only calculates the overlap based
	 * on the centers and the radi, so not the real overlap
	 *
	 * TODO: should we do this by MC to get the real overlap???
	 *
	 * @param other
	 * @return
	 */

	public double overlapRadiusDegree(SphereCluster other) {


		double[] center0 = getCenter();
		double radius0 = getRadius();

		double[] center1 = other.getCenter();
		double radius1 = other.getRadius();

		double radiusBig;
		double radiusSmall;
		if(radius0 < radius1){
			radiusBig = radius1;
			radiusSmall = radius0;
		}
		else{
			radiusBig = radius0;
			radiusSmall = radius1;
		}

		double dist = 0;
		for (int i = 0; i < center0.length; i++) {
			double delta = center0[i] - center1[i];
			dist += delta * delta;
		}
		dist = Math.sqrt(dist);

		if(dist > radiusSmall + radiusBig)
			return 0;
		if(dist + radiusSmall <= radiusBig){
			//one lies within the other
			return 1;
		}
		else{
			return (radiusSmall+radiusBig-dist)/(2*radiusSmall);
		}
	}

	public void combine(SphereCluster cluster) {
		double[] center = getCenter();
		double[] newcenter = new double[center.length];
		double[] other_center = cluster.getCenter();
		double other_weight = cluster.getWeight();
		double other_radius = cluster.getRadius();

		for (int i = 0; i < center.length; i++) {
			newcenter[i] = (center[i]*getWeight()+other_center[i]*other_weight)/(getWeight()+other_weight);
		}
		
		this.center = newcenter;
		double r_0 = getRadius() + Math.abs(distance(center, newcenter));
		double r_1 = other_radius + Math.abs(distance(other_center, newcenter));
		radius = Math.max(r_0, r_1);
		weight+= other_weight;
	}

	public void merge(SphereCluster cluster) {
		double[] c0 = getCenter();
		double w0 = getWeight();
		double r0 = getRadius();

		double[] c1 = cluster.getCenter();
		double w1 = cluster.getWeight();
		double r1 = cluster.getRadius();

		//vector
		double[] v = new double[c0.length];
		//center distance
		double d = 0;

		for (int i = 0; i < c0.length; i++) {
			v[i] = c0[i] - c1[i];
			d += v[i] * v[i];
		}
		d = Math.sqrt(d);



		double r = 0;
		double[] c = new double[c0.length];

		//one lays within the others
		if(d + r0 <= r1  || d + r1 <= r0){
			if(d + r0 <= r1){
				r = r1;
				c = c1;
			}
			else{
				r = r0;
				c = c0;
			}
		}
		else{
			r = (r0 + r1 + d)/2.0;
			for (int i = 0; i < c.length; i++) {
				c[i] = c1[i] - v[i]/d * (r1-r);
			}
		}

		setCenter(c);
		setRadius(r);
		setWeight(w0+w1);

	}

	@Override
	public double[] getCenter() {
		double[] copy = new double[center.length];
		System.arraycopy(center, 0, copy, 0, center.length);
		return copy;
	}

	public void setCenter(double[] center) {
		this.center = center;
	}

	public double getRadius() {
		return radius;
	}

	public void setRadius( double radius ) {
		this.radius = radius;
	}

	@Override
	public double getWeight() {
		return weight;
	}

	public void setWeight( double weight ) {
		this.weight = weight;
	}

	@Override
	public double getInclusionProbability(Instance instance) {
		if (getCenterDistance(instance) <= getRadius()) {
			return 1.0;
		}
		return 0.0;
	}

	public double getCenterDistance(Instance instance) {
		double distance = 0.0;
		//get the center through getCenter so subclass have a chance
		double[] center = getCenter();
		for (int i = 0; i < center.length; i++) {
			double d = center[i] - instance.value(i);
			distance += d * d;
		}
		return Math.sqrt(distance);
	}
	
	/*
	 * the minimal distance from an Instance to the surface of the cluster.
	 * is negative if the Instance is inside the cluster.
	 */
	public double getHullDistance(Instance instance) {
		double distance = getCenterDistance(instance);
		distance -= getRadius();
		return distance;
	}

	public double getCenterDistance(SphereCluster other) {
		return distance(getCenter(), other.getCenter());
	}

	/*
	 * the minimal distance between the surface of two clusters.
	 * is negative if the two clusters overlap
	 */
	public double getHullDistance(SphereCluster other) {
		double distance = 0.0;
		//get the center through getCenter so subclass have a chance
		double[] center0 = getCenter();
		double[] center1 = other.getCenter();
		distance = distance(center0, center1);

		distance = distance - getRadius() - other.getRadius();
		return distance;
	}

	/*
	 */
	/**
	 * When a clusters looses points the new minimal bounding sphere can be
	 * partly outside of the originating cluster. If a another cluster is
	 * right next to the original cluster (without overlapping), the new
	 * cluster can be overlapping with this second cluster. OverlapSave
	 * will tell you if the current cluster can degenerate so much that it
	 * overlaps with cluster 'other'
	 * 
	 * @param other the potentially overlapping cluster
	 * @return true if cluster can potentially overlap
	 */
	public boolean overlapSave(SphereCluster other){
		//use basic geometry to figure out the maximal degenerated cluster
		//comes down to Max(radius *(sin alpha + cos alpha)) which is
		double minDist = Math.sqrt(2)*(getRadius() + other.getRadius());
		double diff = getCenterDistance(other) - minDist;

		if(diff > 0)
			return true;
		else
			return false;
	}

	private double distance(double[] v1, double[] v2){
		double distance = 0.0;
		double[] center = getCenter();
		for (int i = 0; i < center.length; i++) {
			double d = v1[i] - v2[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}

	public double[] getDistanceVector(Instance instance){
		return distanceVector(getCenter(), instance.toDoubleArray());
	}

	public double[] getDistanceVector(SphereCluster other){
		return distanceVector(getCenter(), other.getCenter());
	}

	// v2 - v1
	private double[] distanceVector(double[] v1, double[] v2){
		double[] v = new double[v1.length];
		for (int i = 0; i < v1.length; i++) {
			v[i] = v2[i] - v1[i];
		}
		return v;
	}
	
	public Instance sample_around_target_hypercube(Random random, double[] point) {
		double[] center = getCenter();

		final int dimensions = center.length;
		double radius = getRadius();
		
		double res[] = new double[dimensions];
		
		double[] dim_dis = distanceVector(point, getCenter());
		
		for (int i = 0; i < dimensions; ++i) {
			double left_or_right = random.nextDouble();
			double sign;
			if (left_or_right > 0.5) {
				sign = 1;
			} else if (left_or_right < 0.5) {
				sign = -1;
			} else {
				sign = 0;
			}
			res[i] = point[i] + sign * Math.abs(random.nextGaussian()) * (radius - sign * dim_dis[i]) / 3.0;
			
		}
		
		return new DenseInstance(1.0, res);
	}
	
	public Instance sample_around_target_hypercube_boundsphere(Random random, double[] point) {
		Instance tmp_point = sample_around_target_hypercube(random, point);
		
		double[] tmp_point_vector = tmp_point.toDoubleArray();
		
		while (distance(tmp_point_vector, getCenter()) > this.radius) {
			tmp_point = sample_around_target_hypercube(random, point);
			tmp_point_vector = tmp_point.toDoubleArray();
		}
		
		return tmp_point;	
	}
	
	/**
	 * @param random the random
	 * @param target the target mode of the skewed distribution. This point must be inside this sphere cluster.
	 * @return a point sampled from the skewed distribution with the mode at "target"
	 */
	public Instance sample_around_target(Random random, double[] target) {
		double[] center = getCenter();
		double radius = getRadius();
		
		double target_center_distance = distance(target, center);
		if (target_center_distance > radius) {
			System.err.println("Target is outside this sphere cluster.");
			System.err.println("Target: " + Arrays.toString(target));
			System.err.println("Center: " + Arrays.toString(center));
			System.err.println("Radius of this sphere cluster: " + radius);
			System.err.println("Distance between target and center: " + target_center_distance);
			return null;
		}

		final int dimensions = center.length;
		
		double[] point = sample_hypersphere(random, target, 1.0).toDoubleArray();
		double[] direction_vector = distanceVector(target, point);
		double t_intercept = hypersphere_line_intercept_scaler(target, point);
		
		double t_sample = Math.abs(random.nextGaussian()) * t_intercept / 3.0;
		
		double[] res = new double[dimensions];
		for (int i = 0; i < dimensions; ++i) {
			res[i] = target[i] + t_sample * direction_vector[i];
		}
		
		return new DenseInstance(1.0, res);
	}
	
	/**
	 * @param target the origin of the line
	 * @param another_point_on_line the other point on the same line
	 * @return the scaler t in the parameterised equation of the line: x_i = x_0 + t * v_i
	 */
	private double hypersphere_line_intercept_scaler(double[] target, double[] another_point_on_line) {
		double[] center = getCenter();
		double radius = getRadius();
		
		final int dimensions = center.length;
		
		double[] delta = distanceVector(target, another_point_on_line);
		double[] gamma = distanceVector(target, center);
		
		double a = 0, b = 0, c = 0;
		for (int i = 0; i < dimensions; ++i) {
			a += (delta[i] * delta[i]);
			b += (delta[i] * gamma[i]);
			c += (gamma[i] * gamma[i]);
		}
		
		b *= -2;
		c -= (radius * radius);
		
		double bb4ac = b * b - 4 * a * c;
		
		// We just need the +ve "t", as it goes with the direction vector. i.e. it follows the given angles.
		// The -ve "t" follows the angles oppositely.
		return (-b + Math.sqrt(bb4ac)) / (2 * a);
	}
	
	/**
	 * Samples this cluster by returning a point from inside it.
	 * @param random a random number source
	 * @return a point that usually (99.9% of time) lies inside this cluster and it is likely close to the centre.
	 */
	public Instance sampleGaussian(Random random) {
		double r_sample = Math.abs(random.nextGaussian()) * getRadius() / 3.0;
		
		return sample_hypersphere(random, getCenter(), r_sample);
	}

 
	/**
	 * Samples this cluster by returning a point from inside it.
	 * @param random a random number source
	 * @return a point that lies inside this cluster
	 */
	public Instance sample(Random random) {
		return sample_hypersphere(random, getCenter(), getRadius());
	}
	
	public static Instance sample_hypersphere(Random random, double[] center, double radius) {
		final int dimensions = center.length;
		
		double[] g = new double[dimensions];
		for (int i = 0; i < dimensions; ++i) {
			g[i] = random.nextGaussian();
		}
		
		double l2_norm = 0;
		for (int i = 0; i < g.length; ++i) {
			l2_norm += (g[i] * g[i]);
		}
		l2_norm = Math.sqrt(l2_norm);
		
		for (int i = 0; i < dimensions; ++i) {
			g[i] /= l2_norm;
		}
		
		double u_dRoot = Math.pow(random.nextDouble(), 1.0/dimensions);
		
		for (int i = 0; i < dimensions; ++i) {
			g[i] = radius * g[i] * u_dRoot + center[i];
		}
		
		return new DenseInstance(1.0, g);
	}

	@Override
	protected void getClusterSpecificInfo(ArrayList<String> infoTitle, ArrayList<String> infoValue) {
		super.getClusterSpecificInfo(infoTitle, infoValue);
		infoTitle.add("Radius");
		infoValue.add(Double.toString(getRadius()));
	}


}
