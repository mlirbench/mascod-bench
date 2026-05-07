import requests

VALIDATION_API_URL = "http://old-coms-4020-040.class.las.iastate.edu:8080/nova/validate-mutation"
TORCH_VALIDATION_API_URL = "http://old-coms-4020-040.class.las.iastate.edu:8080/nova/validate-torch"

def validate_mutation(input_mlir: str, mutated_mlir: str):
    try:
        response = requests.post(
            VALIDATION_API_URL,
            json={
                "input_mlir": input_mlir,
                "mutated_mlir": mutated_mlir
            },
            # timeout=10
        )
        return response.json()

    except Exception as e:
        return {"valid": False, "error": str(e)}
    
def validate_torch_mutation(input_mlir: str, mutated_mlir: str):
    try:
        response = requests.post(
            TORCH_VALIDATION_API_URL,
            json={
                "input_mlir": input_mlir,
                "mutated_mlir": mutated_mlir
            },
            # timeout=10
        )
        return response.json()

    except Exception as e:
        return {"valid": False, "error": str(e)}