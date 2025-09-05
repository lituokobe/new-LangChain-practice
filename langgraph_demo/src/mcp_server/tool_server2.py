import json

from fastmcp import FastMCP
from cryptography.hazmat.primitives import serialization
from fastmcp.server.auth.providers.jwt import RSAKeyPair, JWTVerifier
import base64
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
from fastmcp.server.dependencies import AccessToken, get_access_token

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# TODO: Create RSA key pair
key_pair = RSAKeyPair.generate()
# print(key_pair.public_key)

# Configure authentication
# auth = BearerAuthProvider( # This function is depreciated
#     public_key = key_pair.public_key, # public key to authenticate signature (token)
#     issuer = "http://lituokobe.com", # your company's domain name as the authentication issuer
#     audience = "Luis_Miguel_server", # service provider's name
# )

# We will use JWTVerifier to do the authentication instead.
# Below is the code to convert key_pair.public_key to jwks
public_key_obj = serialization.load_pem_public_key(
    key_pair.public_key.encode("utf-8")
)
def rsa_public_key_to_jwk(pub_key, kid="my-key-id"):
    numbers = pub_key.public_numbers()
    n = base64.urlsafe_b64encode(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")).rstrip(b"=")
    e = base64.urlsafe_b64encode(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")).rstrip(b"=")
    return {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": n.decode("utf-8"),
        "e": e.decode("utf-8"),
    }

jwks_data = {
    "keys": [
        rsa_public_key_to_jwk(public_key_obj, kid="my-key-id")
    ]
}

# Write JWKS to a temporary file
jwks_file = "jwks.json"
with open(jwks_file, "w") as f:
    json.dump(jwks_data, f)

auth = JWTVerifier(
    jwks_uri=f"file://{jwks_file}",
    issuer="http://lituokobe.com",
    audience="Luis_Miguel_server",
)

# Simulate to generate a token in server
token = key_pair.create_token(
    subject = "dev_user",
    issuer = "http://lituokobe.com",
    audience = "Luis_Miguel_server",
    scopes = ["digger", "invoke_tools"],
    expires_in_seconds = 3600
)
print(f"Test token : {token}")

luis_miguel_server = FastMCP(name = "my_mcp",
                             instructions = "Luis Miguel's Python MCP",
                             auth = auth) # Authenticate the token

# TODO: Create tool MCP
@luis_miguel_server.tool(name = "Tavily_search_tool") #mcp server tool decorator
def my_search(query : str) -> str:
    """
    tool to search public content on the Internet, including real-time weather
    """
    try:
        print("Use the search tool, input parameter is ", query)

        # get token after authorization. It will be None if not authorized.
        access_token: AccessToken = get_access_token()

        if access_token:
            print("full token", access_token)
            print(access_token.scopes)
        else:
            return "No content found in searching due to no authorization. Please pass valid token to the client."

        search = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)
        response = search.run(query)
        if response["results"]:
            return "\n\n".join([d["content"] for d in response["results"]])
        else:
            return "No content found in searching"
    except Exception as e:
        print(e)
        return "No content found in searching"
#
# @luis_miguel_server.tool()
# def say_hello(username : str) -> str:
#     """
#     Greet the designated user.
#     """
#     return f"Hello, {username}! It's a good day today!"