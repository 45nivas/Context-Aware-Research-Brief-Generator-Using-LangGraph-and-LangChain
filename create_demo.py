#!/usr/bin/env python3
"""
Generate a demo research brief for evaluators to view
"""
import requests
import json
import time

def create_evaluator_demo():
    print("🚀 Creating a completed research brief for evaluators...")
    print()

    # Generate a research brief that evaluators can immediately view
    url = 'http://127.0.0.1:8000/brief'
    data = {
        'topic': 'Impact of AI-powered tutoring systems on personalized learning outcomes in K-12 education',
        'depth': 3,
        'follow_up': False,
        'user_id': 'evaluator_demo',
        'context': 'Focus on classroom implementation, student engagement, and learning effectiveness for academic research'
    }

    print(f"📋 Generating brief for: {data['topic']}")
    print("⏳ This will create a ready-to-view assignment for evaluators...")
    print()

    try:
        response = requests.post(url, json=data, timeout=15)
        if response.status_code == 202:
            result = response.json()
            workflow_id = result.get('workflow_id')
            print(f"✅ Request accepted! Workflow ID: {workflow_id}")
            print()
            print("📊 Monitoring progress...")
            
            # Monitor for completion
            for i in range(24):  # 2 minutes max
                time.sleep(5)
                try:
                    status_response = requests.get(f'http://127.0.0.1:8000/brief/{workflow_id}/status', timeout=5)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status', 'unknown')
                        print(f"Check {i+1}: {status}")
                        
                        if status == 'completed':
                            print()
                            print("🎉 RESEARCH BRIEF COMPLETED!")
                            print("=" * 50)
                            print()
                            print("📋 FOR EVALUATORS:")
                            print(f"1. Open browser: http://127.0.0.1:8000")
                            print(f"2. You will see the completed workflow: {workflow_id}")
                            print(f"3. Click 'FULL BRIEF CONTENT' to view the complete assignment")
                            print(f"4. Direct link: http://127.0.0.1:8000/brief/{workflow_id}/full")
                            print(f"5. Web view: http://127.0.0.1:8000/brief/{workflow_id}/web")
                            print()
                            print("✅ Assignment is ready for evaluation!")
                            return workflow_id
                        elif status == 'failed':
                            print(f"❌ Workflow failed: {status_data.get('error', 'Unknown error')}")
                            return None
                    else:
                        print(f"Status check failed: {status_response.status_code}")
                except Exception as e:
                    print(f"Status check error: {e}")
            else:
                print("⚠️  Still processing... evaluators can check later")
                print(f"Workflow ID: {workflow_id}")
                return workflow_id
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    create_evaluator_demo()
