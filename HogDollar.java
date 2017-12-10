/**
 * Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
 */

package KKH.HogDollar;

public class HogDollar {

    private int binSize;
    private int nOrients;
    private int softBin;
    private int useHog;
    private float clipHog;
    private boolean full_angle;

    static {System.loadLibrary("HogDollarJNI");}

    public HogDollar()
    {
        set_params_dalal_HOG();
    }

    /**
     * Set HOG hyperparameters to be the same as Dalal's version.
     */
    public void set_params_dalal_HOG()
    {
        binSize = 8;
        nOrients = 9;
        clipHog = 0.2f;
        softBin = 1;
        useHog = 1;
        full_angle = false;
    }

    /**
     * Set HOG hyperparameters to be the same as Pedro Falzen's version.
     */
    public void set_params_falzen_HOG()
    {
        binSize = 8;
        nOrients = 9;
        clipHog = 0.2f;
        softBin = -1;
        useHog = 2;
        full_angle = true;
    }

    /**
     * Set bin size for HOG
     * @param binSize_
     */
    public void set_param_binSize(int binSize_)
    {
        binSize = binSize_;
    }

    public void set_params_custom(int binSize_, int nOrients_, float clipHog_, int softBin_, int useHog_, boolean full_angle_)
    {
        binSize = binSize_;
        nOrients = nOrients_;
        clipHog = clipHog_;
        softBin = softBin_;
        useHog = useHog_;
        full_angle = full_angle_;
    }

    /**
     * Get number of channels of HOG feature matrix
     * @return
     */
    public int nchannels_hog()
    {
        return useHog == 0 ? nOrients : (useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
    }

    /**
     * Get number of rows of HOG feature matrix
     * @param nrows_img
     * @return
     */
    public int nrows_hog(int nrows_img)
    {
        return nrows_img / binSize;
    }

    /**
     * Get number of cols of HOG feature matrix
     * @param ncols_img
     * @return
     */
    public int ncols_hog(int ncols_img)
    {
        return ncols_img / binSize;
    }

    /**
     * Extract HOG features.
     * @param img Input image array. Stored in col major.
     * @param nrows_img Number of rows of image
     * @param ncols_img Number of cols of image
     * @param nchannels_img Number of channels of image (either 1 or 3).
     * @return an array of features stored in col major order.
     */
    public float[] extract(float[] img, int nrows_img, int ncols_img, int nchannels_img)
    {
        // compute gradient magnitude and orientation

        float[] M = new float[nrows_img * ncols_img];
        float[] O = new float[nrows_img * ncols_img];
        gradMag(img, M, O, nrows_img, ncols_img, nchannels_img, full_angle);

        // compute gradient histogram

        int nrows_H = nrows_hog(nrows_img);
        int ncols_H = ncols_hog(ncols_img);
        int nchannels_H = nchannels_hog();
        float[] H = new float[nrows_H * ncols_H * nchannels_H];

        if (useHog == 0)
            gradHist(M, O, H, nrows_img, ncols_img, binSize, nOrients, softBin, full_angle);
	    else if (useHog == 1)
            hog(M, O, H, nrows_img, ncols_img, binSize, nOrients, softBin, full_angle, clipHog);
	    else
            fhog(M, O, H, nrows_img, ncols_img, binSize, nOrients, softBin, clipHog);

	    return H;
    }

    private native void gradMag(float[] img, float[] M, float[] O, int nrows_img, int ncols_img, int nchannels_img, boolean full_angle);
    private native void gradHist(float[] M, float[] O, float[] H, int nrows_img, int ncols_img, int binSize, int nOrients, int softBin, boolean full_angle);
    private native void hog(float[] M, float[] O, float[] H, int nrows_img, int ncols_img, int binSize, int nOrients, int softBin, boolean full_angle, float clipHog);
    private native void fhog(float[] M, float[] O, float[] H, int nrows_img, int ncols_img, int binSize, int nOrients, int softBin, float clipHog);

}
