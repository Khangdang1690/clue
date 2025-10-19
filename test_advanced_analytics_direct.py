"""
Direct test of Advanced Analytics integration (bypasses autonomous explorer).

This test loads the data that was already created and stored, then directly
runs the advanced analytics nodes to demonstrate forecasting, anomaly detection,
causal inference, and variance decomposition.
"""

import pandas as pd
from sqlalchemy.orm import Session
from src.database.connection import DatabaseManager
from src.database.repository import DatasetRepository, CompanyRepository
from src.database.models import Company, Dataset

# Import advanced analytics modules
from src.analytics import (
    TimeSeriesForecaster,
    AnomalyDetector,
    CausalAnalyzer,
    VarianceDecomposer,
    BusinessInsightSynthesizer
)

def main():
    print("=" * 80)
    print("DIRECT ADVANCED ANALYTICS TEST")
    print("=" * 80)
    print("\nThis test demonstrates advanced analytics on existing data:")
    print("  1. Time series forecasting (revenue prediction)")
    print("  2. Anomaly detection (unusual spikes/drops)")
    print("  3. Causal analysis (marketing -> revenue)")
    print("  4. Variance decomposition (what drives churn)")
    print("=" * 80)

    with DatabaseManager.get_session() as session:
        # Find the company
        company = session.query(Company).filter_by(
            name="Advanced Analytics Demo"
        ).first()

        if not company:
            print("\n[ERROR] Company 'Advanced Analytics Demo' not found!")
            print("Please run test_etl_advanced_analytics.py first to create the data.")
            return

        print(f"\n[FOUND] Company: {company.name} (ID: {company.id})")

        # Load datasets
        datasets = session.query(Dataset).filter_by(
            company_id=company.id
        ).all()

        print(f"[FOUND] {len(datasets)} datasets")
        for ds in datasets:
            print(f"  - {ds.table_name}: {ds.row_count} rows")

        # Find the sales_performance dataset (time series data)
        sales_dataset = next((d for d in datasets if 'sales_performance' in d.table_name), None)
        customer_dataset = next((d for d in datasets if 'customer_segments' in d.table_name), None)

        if not sales_dataset:
            print("\n[ERROR] sales_performance dataset not found!")
            return

        # Load the actual data
        print(f"\n[LOADING] {sales_dataset.table_name}...")
        sales_df = DatasetRepository.load_dataframe(session, sales_dataset.id)
        print(f"[OK] Loaded {len(sales_df)} rows × {len(sales_df.columns)} columns")
        print(f"Columns: {list(sales_df.columns)}")

        # Initialize synthesizer
        synthesizer = BusinessInsightSynthesizer()

        print("\n" + "=" * 80)
        print("TEST 1: TIME SERIES FORECASTING")
        print("=" * 80)

        # Convert to time series (use index as date if not already datetime)
        if 'date' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            sales_df = sales_df.set_index('date').sort_index()

        if isinstance(sales_df.index, pd.DatetimeIndex):
            print(f"[OK] Time series detected: {len(sales_df)} observations")
            print(f"Date range: {sales_df.index.min()} to {sales_df.index.max()}")

            if 'revenue' in sales_df.columns and len(sales_df) >= 12:
                print("\n[RUNNING] Forecasting revenue for next 6 periods...")
                forecaster = TimeSeriesForecaster(method='auto')

                try:
                    forecast_result = forecaster.forecast(
                        data=sales_df['revenue'],
                        periods=6,
                        context={
                            'metric_name': 'Monthly Revenue',
                            'dataset_name': 'sales_performance'
                        }
                    )

                    print(f"\n[RESULT] Forecast generated")
                    print(f"  Method used: {forecast_result.results.get('model_name', 'unknown')}")
                    predictions = forecast_result.results.get('predictions', [])
                    print(f"  Predictions: {len(predictions)} periods")
                    print(f"  Confidence: {forecast_result.validation.confidence_level}")

                    # Show first few predictions
                    for i, pred in enumerate(predictions[:3]):
                        print(f"    Period {i+1}: ${pred:,.0f}")

                    # Generate business insight
                    print("\n[SYNTHESIZING] Business insight...")
                    insight = synthesizer.synthesize_forecast(forecast_result)
                    print("\n" + "=" * 80)
                    print(insight)
                    print("=" * 80)

                except Exception as e:
                    print(f"[ERROR] Forecasting failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("[SKIP] Not enough data for forecasting (need 12+ observations)")
        else:
            print("[SKIP] No datetime index found")

        print("\n" + "=" * 80)
        print("TEST 2: ANOMALY DETECTION")
        print("=" * 80)

        if 'revenue' in sales_df.columns:
            print(f"\n[RUNNING] Detecting anomalies in revenue...")
            detector = AnomalyDetector(method='auto')

            try:
                anomaly_result = detector.detect_anomalies(
                    data=sales_df['revenue'],
                    context={
                        'metric_name': 'Monthly Revenue',
                        'dataset_name': 'sales_performance'
                    }
                )

                print(f"\n[RESULT] Anomaly detection complete")
                print(f"  Method used: {anomaly_result.results.get('detection_method', 'unknown')}")
                anomalies = anomaly_result.results.get('anomalies', [])
                print(f"  Anomalies found: {len(anomalies)}")
                print(f"  Confidence: {anomaly_result.validation.confidence_level}")

                # Show top anomalies
                if anomalies:
                    print("\n  Top anomalies:")
                    for anomaly in anomalies[:5]:
                        print(f"    {anomaly.get('timestamp', 'unknown')}: ${anomaly['value']:,.0f} "
                              f"(severity: {anomaly['severity']}, score: {anomaly.get('anomaly_score', 0):.2f})")

                    # Generate business insight
                    print("\n[SYNTHESIZING] Business insight...")
                    insight = synthesizer.synthesize_anomalies(anomaly_result)
                    print("\n" + "=" * 80)
                    print(insight)
                    print("=" * 80)
                else:
                    print("  No significant anomalies detected")

            except Exception as e:
                print(f"[ERROR] Anomaly detection failed: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("TEST 3: CAUSAL INFERENCE")
        print("=" * 80)

        if 'marketing_spend' in sales_df.columns and 'revenue' in sales_df.columns:
            print(f"\n[RUNNING] Testing if marketing_spend causes revenue...")
            analyzer = CausalAnalyzer(max_lag=6)

            try:
                causal_result = analyzer.analyze_causality(
                    cause=sales_df['marketing_spend'],
                    effect=sales_df['revenue'],
                    context={
                        'cause_name': 'Marketing Spend',
                        'effect_name': 'Revenue',
                        'dataset_name': 'sales_performance'
                    }
                )

                print(f"\n[RESULT] Causal analysis complete")
                relationships = causal_result.results.get('relationships', [])
                has_causal_effect = any(r['is_significant'] for r in relationships)
                print(f"  Causal relationship: {has_causal_effect}")

                if relationships:
                    rel = relationships[0]  # Primary relationship
                    print(f"  Optimal lag: {rel.get('lag', 0)} periods")
                    print(f"  Strength: {rel.get('strength', 'unknown')}")

                print(f"  Confidence: {causal_result.validation.confidence_level}")

                if has_causal_effect:
                    # Generate business insight
                    print("\n[SYNTHESIZING] Business insight...")
                    insight = synthesizer.synthesize_causal(causal_result)
                    print("\n" + "=" * 80)
                    print(insight)
                    print("=" * 80)
                else:
                    print("  No significant causal relationship found")

            except Exception as e:
                print(f"[ERROR] Causal analysis failed: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("TEST 4: VARIANCE DECOMPOSITION")
        print("=" * 80)

        if customer_dataset:
            print(f"\n[LOADING] {customer_dataset.table_name}...")
            customer_df = DatasetRepository.load_dataframe(session, customer_dataset.id)
            print(f"[OK] Loaded {len(customer_df)} rows × {len(customer_df.columns)} columns")

            # Check if we have churn and feature columns
            if 'churned' in customer_df.columns:
                print(f"\n[RUNNING] Analyzing what drives customer churn...")

                # Select numeric features
                feature_cols = ['engagement_score', 'support_tickets', 'monthly_spend']
                available_features = [col for col in feature_cols if col in customer_df.columns]

                if available_features:
                    X = customer_df[available_features]
                    y = customer_df['churned']

                    decomposer = VarianceDecomposer(method='auto')

                    try:
                        variance_result = decomposer.decompose(
                            X=X,
                            y=y,
                            context={
                                'outcome': 'Customer Churn',
                                'dataset_name': 'customer_segments'
                            }
                        )

                        print(f"\n[RESULT] Variance decomposition complete")
                        total_variance = variance_result.results.get('total_variance_explained', 0)
                        print(f"  Total variance explained: {total_variance:.2%}")
                        print(f"  Confidence: {variance_result.validation.confidence_level}")

                        # Show top contributors
                        contributions = variance_result.results.get('feature_contributions', [])
                        print("\n  Top contributors to churn:")
                        for contrib in contributions[:5]:
                            print(f"    {contrib['feature']}: {contrib['contribution']:.2%}")

                        # Generate business insight
                        print("\n[SYNTHESIZING] Business insight...")
                        insight = synthesizer.synthesize_variance_decomposition(variance_result)
                        print("\n" + "=" * 80)
                        print(insight)
                        print("=" * 80)

                    except Exception as e:
                        print(f"[ERROR] Variance decomposition failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[SKIP] Required feature columns not found")
            else:
                print("[SKIP] No 'churned' column found in customer data")
        else:
            print("[SKIP] Customer dataset not found")

        print("\n" + "=" * 80)
        print("ADVANCED ANALYTICS TEST COMPLETE")
        print("=" * 80)
        print("\nAll tests completed!")
        print("\nNote: The insights above are generated by LLMs from statistical results.")
        print("NO raw statistics (p-values, F-statistics, etc.) are shown to users.")
        print("=" * 80)


if __name__ == "__main__":
    main()
