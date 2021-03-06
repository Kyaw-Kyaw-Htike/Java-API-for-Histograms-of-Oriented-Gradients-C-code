/* Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved. */
#include <jni.h>
/* Header for class KKH_HogDollar_HogDollar */

#ifndef _Included_KKH_HogDollar_HogDollar
#define _Included_KKH_HogDollar_HogDollar
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     KKH_HogDollar_HogDollar
 * Method:    gradMag
 * Signature: ([F[F[FIIIZ)V
 */
JNIEXPORT void JNICALL Java_KKH_HogDollar_HogDollar_gradMag
  (JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint, jint, jint, jboolean);

/*
 * Class:     KKH_HogDollar_HogDollar
 * Method:    gradHist
 * Signature: ([F[F[FIIIIIZ)V
 */
JNIEXPORT void JNICALL Java_KKH_HogDollar_HogDollar_gradHist
  (JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint, jint, jint, jint, jint, jboolean);

/*
 * Class:     KKH_HogDollar_HogDollar
 * Method:    hog
 * Signature: ([F[F[FIIIIIZF)V
 */
JNIEXPORT void JNICALL Java_KKH_HogDollar_HogDollar_hog
  (JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint, jint, jint, jint, jint, jboolean, jfloat);

/*
 * Class:     KKH_HogDollar_HogDollar
 * Method:    fhog
 * Signature: ([F[F[FIIIIIF)V
 */
JNIEXPORT void JNICALL Java_KKH_HogDollar_HogDollar_fhog
  (JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint, jint, jint, jint, jint, jfloat);

#ifdef __cplusplus
}
#endif
#endif
