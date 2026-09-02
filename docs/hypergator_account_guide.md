# FSU CS / RAI HiPerGator 账号申请指南

最后核对日期：2026-09-02

本文面向需要申请 HiPerGator 账号、或需要加入 FSU CS / RAI 相关 HiPerGator group/allocation 的学生、研究人员和合作者。若 UF Research Computing 官方说明更新，以官方页面为准。

## 一句话流程

大多数 FSU 用户应走 federated account 路径：

1. 先确认自己应该由哪个 HiPerGator sponsor/group 批准。
2. 在本机生成 SSH public key。
3. 打开 UF 官方 HiPerGator account request 页面。
4. 选择 Federated Account Request，用 FSU 账号登录。
5. 在表单里填写个人信息、机构、sponsor/group 和简短说明。
6. 提交后查收邮件，确认邮箱、同意条款，并上传 `.pub` SSH public key 文件。
7. 等待 sponsor 批准和 UFRC 创建账号。
8. 账号开通后验证登录，并让 group manager 加入需要的项目或部门 allocation。

已经有 HiPerGator 账号时，不要重复申请新账号；需要加 group/allocation 时应走 UFRC support ticket 或 group manager 流程。

## 该选哪一种申请入口

官方入口：

https://it.ufl.edu/rc/get-started/request-hipergator-account/

FSU 用户通常选择 `Federated Account Request`。UFRC 的 federated access 面向 InCommon federation 机构，FSU 已在 UFRC federated liaison 列表中。

只有已经拥有 UF GatorLink 的用户才走 `UF Account Request`。非 UF、且不能走 federated login 的外部合作者，通常需要先申请 affiliate GatorLink，再用 UF Account Request 申请 HiPerGator；官方说明提到繁忙时期 GatorLink 创建可能需要最多两周。

## 申请前准备

请先准备好：

- FSU institutional email。
- Organization：`Florida State University`。
- 正确的 HiPerGator sponsor 或 group 名称。
- SSH public key 文件，通常是 `~/.ssh/id_ed25519_hipergator.pub`。
- 一句说明为什么需要 access。

如果不知道 sponsor/group 名称，不要猜。先问 PI、lab manager 或 FSU CS HiPerGator group manager。普通用户的账号申请需要已有 sponsor/group/resource allocation 支持；如果 sponsor 还没有 HiPerGator sponsor record、primary group 和资源，普通用户账号不会被创建。

## 生成 SSH Public Key

Linux/macOS:

```bash
ssh-keygen -t ed25519 -C "your_fsu_email@fsu.edu" -f ~/.ssh/id_ed25519_hipergator
cat ~/.ssh/id_ed25519_hipergator.pub
```

Windows PowerShell:

```powershell
ssh-keygen -t ed25519 -C "your_fsu_email@fsu.edu" -f $env:USERPROFILE\.ssh\id_ed25519_hipergator
type $env:USERPROFILE\.ssh\id_ed25519_hipergator.pub
```

保管好 private key，不要分享没有 `.pub` 后缀的私钥文件。Federated account request 流程会要求准备并上传 public key 文件，即使你之后主要使用 Open OnDemand、JupyterHub 或 Galaxy 等网页入口。

## 提交账号申请

1. 打开官方申请页：
   https://it.ufl.edu/rc/get-started/request-hipergator-account/
2. 点击 `Federated Account Request`。
3. 选择 `Florida State University`，然后用 FSU credential 登录。
4. 在表单里填写：
   - Given Name
   - Family Name
   - Email
   - Organization，例如 `Florida State University`
   - Sponsor，从下拉框选择正确 sponsor
   - Comments，写一两句用途说明
5. 提交表单。官方文档提醒：整个 request form 需要在同一个浏览器 session 中完成；如果关闭或超时，可能需要联系 UFRC support 重置。
6. 查收邮件，点击确认链接；如果没收到，检查 spam/junk。
7. 阅读 Terms and Conditions，选择同意。
8. 选择并上传准备好的 `.pub` SSH public key 文件。

可用 comments 模板：

```text
I am an FSU CS / RAI researcher requesting HiPerGator access for project work under the relevant FSU CS sponsored group. Please add me to the appropriate group/allocation after sponsor approval.
```

上传 SSH public key 后，UFRC 会通知 sponsor 审批。账号不会在 sponsor 确认前创建。官方说明中，sponsor 确认后账号创建通常需要 2-3 个工作日。

## 账号创建后

先完成 HiPerGator 要求的新用户 training/onboarding，然后验证登录。

SSH host 是：

```bash
ssh -i ~/.ssh/id_ed25519_hipergator YOUR_HIPERGATOR_USERNAME@hpg.rc.ufl.edu
```

注意：

- Federated 用户的 HiPerGator username 可能不是 FSU email 前缀，以开通邮件为准。
- UFRC 文档说明 federated 用户 SSH login 通常需要 SSH key；SSH 到 `hpg.rc.ufl.edu` 使用 port `22`，federated 用户可能需要先连接 eduVPN。
- 如果 SSH 失败，检查 private key 路径、网络/VPN、username，以及是否已完成账号创建邮件里的全部步骤。

登录后确认：

- 默认 group 是哪个。
- Slurm job 应该 charge 到哪个 account/allocation。
- 项目数据应该放在哪个 storage path。
- 自己是否需要 FSU CS umbrella compute allocation、PI 自己的 group，或两者都需要。

## FSU CS 部门 Allocation 背景摘要

附件中的 2026 年 1-3 月邮件线程讨论的是 FSU CS department-level umbrella investment，用于多个 FSU CS research groups 共享，而不是某一个人的个人账号申请。

邮件线程中更新后的配置为：

- 128 NCU
- 20 BSU
- 8 OSU
- 24 NGU
- 总报价：USD 106,300

UFRC 在邮件中明确过这些操作要点：

- quote/purchase 解决资源采购，不等于自动给所有用户开户。
- umbrella group 仍需要 sponsor/investor 记录；系统里必须关联真实人员，即使联系邮箱可以用 departmental/shared email。
- shared email 可作为 primary/secondary contact，但自动确认邮件可能要求从 UFRC 数据库里的邮箱回复；若 distribution list 不能直接回复，就需要到 support.rc.ufl.edu 的 ticket 里 comment。
- compute allocation 适合放入 department umbrella group。
- storage 不建议全部放入 umbrella group，因为共享 storage 容易被个别 group 用满并影响所有人；更稳妥的做法通常是 compute 放 umbrella，storage 分给 individual sponsored groups。
- 为了减少每次加用户都单独确认，部门应维护一份有资格加入 umbrella group 的 FSU sponsors/groups 列表，并明确谁负责审批新用户。

## RAI / FSU CS 推荐内部流程

新 lab member：

1. Lab 先确认用户属于哪个 PI/sponsor，以及是否需要 individual group、FSU CS umbrella allocation，或两者。
2. 用户提交 federated HiPerGator account request，并选择正确 sponsor。
3. Sponsor 或指定 group manager 审批。
4. 账号创建后，group manager 把该用户加入需要的 group/allocation。
5. 用户验证 SSH 或 Open OnDemand 登录，再运行一个小型 Slurm smoke test。

新 PI/group 想使用 FSU CS umbrella allocation：

1. 先确认部门允许该 group 使用共享 allocation。
2. 联系 umbrella group manager 或 UFRC support，把 PI/group 关联到 umbrella group。
3. 单独决定 storage 是否留在 PI 自己的 group，而不是放进 umbrella group。
4. 记录该 group 中谁有权审批未来用户。

## 常见问题

Sponsor 不在列表里：

- 不要随便选一个 sponsor。
- 先问 PI/group manager 确认准确名称。
- 如果这是新 group，可能需要 sponsor 先申请 sponsor-level HiPerGator account 和资源。

申请卡住：

- 检查 spam/junk。
- 问 sponsor 是否已经收到并批准确认请求。
- 开 UFRC support ticket，写清姓名、institutional email、提交日期和 sponsor/group。

需要加入额外 allocation：

- 不要重新申请账号。
- 让 allocation owner/group manager 把已有 HiPerGator username 加到对应 group。

## 官方参考

- Request HiPerGator Account: https://it.ufl.edu/rc/get-started/request-hipergator-account/
- Federated Account Request: https://docs.rc.ufl.edu/access/federated_request/
- Federated Account Liaisons: https://docs.rc.ufl.edu/access/federated_liaisons/
- From Zero to HiPerGator: https://docs.rc.ufl.edu/quickstart/zero_hipergator/
- Connecting with SSH: https://docs.rc.ufl.edu/interfaces/terminal/
- Using SSH Keys: https://docs.rc.ufl.edu/access/ssh_keys/
- HiPerGator Quotes and Purchases: https://docs.rc.ufl.edu/resources/quote/
- UFRC Support: https://support.rc.ufl.edu/
