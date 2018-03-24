temp <- prop.table(table(base_test$Label, base_test$age), 2)
prop.table(table(base_test$Label, base_test$education), 2)
prop.table(table(base_test$race, base_test$education), 1)
prop.table(table(base_test$education, base_test$Label, base_test$sex), 1)
