# 创建决策树分类器
# clf = DecisionTreeClassifier(random_state=42)

# # 使用网格搜索进行超参数调优
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # 打印最佳参数和最佳得分
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print(f'Best Parameters: {best_params}')
# print(f'Best Score: {best_score:.4f}')

# # 使用最佳参数重新训练模型
# best_clf = grid_search.best_estimator_
# best_clf.fit(X_train, y_train)

# # 在验证集上评估模型
# y_pred = best_clf.predict(X_val)
# accuracy = accuracy_score(y_val, y_pred)
# print(f'Validation Accuracy with Best Parameters: {accuracy:.4f}')

# # 对测试集进行预测
# test_pred = best_clf.predict(test_data)

# # 将预测结果填入submission.csv中
# submission['Survived'] = test_pred

# # 保存结果到submission.csv
# submission.to_csv('D:\\dataenclorse\\forth\\submission.csv', index=False)

# # 可视化最佳决策树并保存为图片
# dot_data = export_graphviz(
#     best_clf,
#     out_file=None,
#     feature_names=X.columns,
#     class_names=['Not Survived', 'Survived'],
#     filled=True,
#     rounded=True,
#     special_characters=True
# )

# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('D:\\dataenclorse\\forth\\decision_tree_best.png')