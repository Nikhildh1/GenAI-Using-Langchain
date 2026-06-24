from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool


searchTool=DuckDuckGoSearchRun()

result=searchTool.invoke("Who won IPL 2026")

# print(result)

shellTool=ShellTool()

ans=shellTool.invoke('ls')

print(ans)